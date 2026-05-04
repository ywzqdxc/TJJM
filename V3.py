"""
E/S/C三维分量逐年统计补算脚本
================================
从已有的 Vuln_TOPSIS_{year}.npy 和原始指标文件重新计算
每一年的 E（暴露度）、S（敏感性）、C（应对不足）分量均值，
输出表格和图像，对应论文表16与图9。

依赖（须已运行 step3 + step3_annual_module）：
  - VSC_Weights.csv               组合权重
  - Dynamic/Precipitation_{yr}.npy 逐年降雨量
  - Dynamic/CR_{yr}.npy            逐年径流系数
  - Static/twi.npy / hand.npy      静态地形
  - External/*.tif                 静态社会指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import rasterio
from rasterio.warp import reproject, Resampling
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
STATIC_DIR = r'./Step_New/Static'
DYN_DIR    = r'./Step_New/Dynamic'
EXT_DIR    = r'./Step_New/External'
RISK_DIR   = r'./Step_New/Risk_Map'
OUTPUT_DIR = r'./Step_New/Visualization/Step3_VSC'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STUDY_YEARS = list(range(2012, 2025))

# ============================================================
# 工具函数（与step3保持一致）
# ============================================================

def load_npy(path):
    if not os.path.exists(path): return None
    return np.load(path).astype(np.float32)

def load_tif(path, h, w, dst_crs, dst_transform):
    if not os.path.exists(path): return None
    with rasterio.open(path) as src:
        if src.height == h and src.width == w and src.crs == dst_crs:
            arr = src.read(1).astype(np.float32)
            nd = src.nodata
            if nd is not None: arr = np.where(arr == nd, np.nan, arr)
        else:
            arr = np.full((h, w), np.nan, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=arr,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
    return np.where(arr < -1e10, np.nan, arr)

def minmax_norm_fixed(arr, lo, hi, valid_mask, direction='positive'):
    """固定边界的Min-Max归一化（保证年际可比）"""
    result = np.full_like(arr, np.nan, dtype=np.float32)
    if hi - lo < 1e-10:
        result[valid_mask & np.isfinite(arr)] = 0.5
        return result
    arr_safe = np.where(arr == 0, 1e-6, arr)
    normed   = np.clip((arr_safe - lo) / (hi - lo), 0.0, 1.0)
    if direction == 'negative': normed = 1.0 - normed
    result[valid_mask & np.isfinite(arr)] = normed[valid_mask & np.isfinite(arr)]
    return result

# ============================================================
# 一、加载基准信息与权重
# ============================================================
print("[1/4] 加载基准信息与权重...")

with rasterio.open(DEM_PATH) as ref:
    h, w          = ref.height, ref.width
    dst_crs       = ref.crs
    dst_transform = ref.transform

nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

# 加载组合权重
df_w    = pd.read_csv(os.path.join(RISK_DIR, 'VSC_Weights.csv'))
w_combo = df_w['组合'].values.astype(np.float64)

# 维度内归一化权重
w_E = w_combo[0:5] / w_combo[0:5].sum()
w_S = w_combo[5:7] / w_combo[5:7].sum()
w_C = w_combo[7:10] / w_combo[7:10].sum()

print(f"  w_E（E维度内权重）: {[f'{v:.4f}' for v in w_E]}")
print(f"  w_S: {[f'{v:.4f}' for v in w_S]}")
print(f"  w_C: {[f'{v:.4f}' for v in w_C]}")

# ============================================================
# 二、加载静态指标（8个，不随年份变化）
# ============================================================
print("\n[2/4] 加载静态指标...")

static_raw = {
    'twi':  load_npy(os.path.join(STATIC_DIR, 'twi.npy')),
    'hand': load_npy(os.path.join(STATIC_DIR, 'hand.npy')),
    'wp':   load_tif(os.path.join(EXT_DIR, 'waterlogging_point_density_30m.tif'),
                     h, w, dst_crs, dst_transform),
    'pop':  load_tif(os.path.join(EXT_DIR, 'population_density_30m.tif'),
                     h, w, dst_crs, dst_transform),
    'road': load_tif(os.path.join(EXT_DIR, 'road_density_30m.tif'),
                     h, w, dst_crs, dst_transform),
    'sh':   load_tif(os.path.join(EXT_DIR, 'shelter_density_30m.tif'),
                     h, w, dst_crs, dst_transform),
    'ho':   load_tif(os.path.join(EXT_DIR, 'hospital_density_30m.tif'),
                     h, w, dst_crs, dst_transform),
    'fs':   load_tif(os.path.join(EXT_DIR, 'firestation_density_30m.tif'),
                     h, w, dst_crs, dst_transform),
}

for k, v in static_raw.items():
    status = f"有效={np.sum(valid_mask & np.isfinite(v)):,}" if v is not None else "缺失"
    print(f"  {k:6s}: {status}")

# ============================================================
# 三、计算全局归一化边界（跨所有年份，保证年际可比）
# ============================================================
print("\n[3/4] 计算全局归一化边界...")

# 动态指标：汇总所有年份的降雨量和CR
all_rain, all_cr = [], []
year_dyn = {}

rain_mean_fb = load_npy(os.path.join(DYN_DIR, 'Precipitation_Mean.npy'))
cr_mean_fb   = load_npy(os.path.join(DYN_DIR, 'CR_MultiYear_Mean.npy'))

for yr in STUDY_YEARS:
    r = load_npy(os.path.join(DYN_DIR, f'Precipitation_{yr}.npy'))
    c = load_npy(os.path.join(DYN_DIR, f'CR_{yr}.npy'))
    if r is None:
        r = rain_mean_fb
        if yr == STUDY_YEARS[0]: print("  ⚠️  无逐年降雨文件，用多年均值代替")
    if c is None:
        c = cr_mean_fb
    year_dyn[yr] = {'rain': r, 'cr': c}
    if r is not None: all_rain.append(r[valid_mask & np.isfinite(r)])
    if c is not None: all_cr.append(c[valid_mask & np.isfinite(c)])

# 全局边界（P0.5~P99.5，避免极端值拉伸）
rain_all = np.concatenate(all_rain) if all_rain else np.array([200, 800])
cr_all   = np.concatenate(all_cr)   if all_cr   else np.array([0.5, 0.95])

BOUNDS = {
    'rain_lo': float(np.percentile(rain_all, 0.5)),
    'rain_hi': float(np.percentile(rain_all, 99.5)),
    'cr_lo':   float(cr_all.min()),
    'cr_hi':   float(cr_all.max()),
}

# 静态指标边界（基于全域有效像元）
def static_bound(arr, plo=0.5, phi=99.5):
    v = arr[valid_mask & np.isfinite(arr)]
    return float(np.percentile(v, plo)), float(np.percentile(v, phi))

BOUNDS_S = {k: static_bound(v) for k, v in static_raw.items() if v is not None}

print(f"  降雨量全局边界: [{BOUNDS['rain_lo']:.1f}, {BOUNDS['rain_hi']:.1f}] mm")
print(f"  CR全局边界:     [{BOUNDS['cr_lo']:.4f}, {BOUNDS['cr_hi']:.4f}]")

# ============================================================
# 四、逐年计算 E/S/C 分量均值
# ============================================================
print("\n[4/4] 逐年计算E/S/C分量...")

# 静态指标归一化（固定边界，全年一致）
def sn(key, dire):
    arr = static_raw.get(key)
    if arr is None:
        return np.where(valid_mask, 0.0, np.nan).astype(np.float32)
    lo, hi = BOUNDS_S.get(key, (0.0, 1.0))
    return minmax_norm_fixed(arr, lo, hi, valid_mask, dire)

twi_n  = sn('twi',  'positive')
hand_n = sn('hand', 'negative')   # HAND逆向：低洼=高风险
wp_n   = sn('wp',   'positive')
pop_n  = sn('pop',  'positive')
road_n = sn('road', 'positive')
sh_n   = sn('sh',   'negative')   # 应对能力逆向
ho_n   = sn('ho',   'negative')
fs_n   = sn('fs',   'negative')

# 构建静态部分有效掩膜
static_norm_list = [twi_n, hand_n, wp_n, pop_n, road_n, sh_n, ho_n, fs_n]
static_valid = valid_mask.copy()
for arr in static_norm_list:
    static_valid &= np.isfinite(arr)

rows = []
for yr in STUDY_YEARS:
    dyn = year_dyn[yr]
    rain_yr = dyn['rain']
    cr_yr   = dyn['cr']

    # 动态指标归一化（固定全局边界）
    rain_n = minmax_norm_fixed(rain_yr, BOUNDS['rain_lo'], BOUNDS['rain_hi'],
                                valid_mask, 'positive') if rain_yr is not None \
             else np.where(valid_mask, 0.0, np.nan).astype(np.float32)
    cr_n   = minmax_norm_fixed(cr_yr, BOUNDS['cr_lo'], BOUNDS['cr_hi'],
                                valid_mask, 'positive') if cr_yr is not None \
             else np.where(valid_mask, 0.0, np.nan).astype(np.float32)

    # 完整有效像元（动态+静态均有效）
    px = static_valid & np.isfinite(rain_n) & np.isfinite(cr_n)
    n  = int(px.sum())
    if n == 0:
        print(f"  {yr}: 无有效像元，跳过")
        continue

    # 提取各维度归一化指标（顺序：R, CR, TWI, HAND, WP | PD, RD | SH, HO, FS）
    E_arr = np.column_stack([rain_n[px], cr_n[px], twi_n[px], hand_n[px], wp_n[px]])
    S_arr = np.column_stack([pop_n[px], road_n[px]])
    C_arr = np.column_stack([sh_n[px], ho_n[px], fs_n[px]])

    # 维度得分（各指标已归一化至[0,1]，维度内加权平均）
    E_score = (E_arr @ w_E)   # shape (n,)
    S_score = (S_arr @ w_S)
    C_score = (C_arr @ w_C)

    # 统计：均值、标准差、P5、P50、P95
    def stats(arr):
        return {
            'mean': float(arr.mean()),
            'std':  float(arr.std()),
            'p5':   float(np.percentile(arr, 5)),
            'p25':  float(np.percentile(arr, 25)),
            'p50':  float(np.percentile(arr, 50)),
            'p75':  float(np.percentile(arr, 75)),
            'p95':  float(np.percentile(arr, 95)),
        }

    e_st = stats(E_score)
    s_st = stats(S_score)
    c_st = stats(C_score)

    # 同步加载综合脆弱性均值（从Annual_Vuln_Stats.csv）
    rows.append({
        '年份': yr,
        'E均值': e_st['mean'], 'E标准差': e_st['std'],
        'E_P5': e_st['p5'],   'E_P50': e_st['p50'], 'E_P95': e_st['p95'],
        'S均值': s_st['mean'], 'S标准差': s_st['std'],
        'S_P5': s_st['p5'],   'S_P50': s_st['p50'], 'S_P95': s_st['p95'],
        'C均值': c_st['mean'], 'C标准差': c_st['std'],
        'C_P5': c_st['p5'],   'C_P50': c_st['p50'], 'C_P95': c_st['p95'],
        '有效像元数': n,
    })

    print(f"  {yr}: E={e_st['mean']:.4f}  S={s_st['mean']:.4f}  "
          f"C={c_st['mean']:.4f}  (n={n:,})")

# ============================================================
# 整理输出
# ============================================================
df_esc = pd.DataFrame(rows)

# 保存完整逐年统计CSV
out_csv = os.path.join(RISK_DIR, 'Annual_ESC_Stats.csv')
df_esc.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n  Annual_ESC_Stats.csv saved ({len(df_esc)} 行)")

# ──────────────────────────────────────────────────
# 生成论文表格（简洁版，对应论文表16）
# ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("论文表16：2012—2024年E/S/C三因子分量统计")
print("=" * 60)
print(f"  {'年份':6s}  {'E均值':>8}  {'S均值':>8}  {'C均值':>8}")
print("  " + "-"*36)
for _, row in df_esc.iterrows():
    print(f"  {int(row['年份']):6d}  {row['E均值']:>8.4f}  {row['S均值']:>8.4f}  {row['C均值']:>8.4f}")

# 变化量汇总（首尾年对比）
yr_first = df_esc.iloc[0]
yr_last  = df_esc.iloc[-1]
print(f"\n  {'':6s}  {'E变化':>8}  {'S变化':>8}  {'C变化':>8}")
dE = yr_last['E均值'] - yr_first['E均值']
dS = yr_last['S均值'] - yr_first['S均值']
dC = yr_last['C均值'] - yr_first['C均值']
print(f"  {'变化量':6s}  {dE:>+8.4f}  {dS:>+8.4f}  {dC:>+8.4f}")
print(f"  首年({int(yr_first['年份'])}) E={yr_first['E均值']:.4f}  S={yr_first['S均值']:.4f}  C={yr_first['C均值']:.4f}")
print(f"  末年({int(yr_last['年份'])}) E={yr_last['E均值']:.4f}  S={yr_last['S均值']:.4f}  C={yr_last['C均值']:.4f}")
print("=" * 60)

# ──────────────────────────────────────────────────
# 绘图：图9三因子分量时序演变图
# ──────────────────────────────────────────────────
years = df_esc['年份'].values
E_m = df_esc['E均值'].values
S_m = df_esc['S均值'].values
C_m = df_esc['C均值'].values
E_s = df_esc['E标准差'].values
S_s = df_esc['S标准差'].values
C_s = df_esc['C标准差'].values

fig, ax = plt.subplots(figsize=(13, 7), facecolor='white')

# 颜色与样式
cfg = {
    'E': {'color': '#E74C3C', 'marker': 'o', 'ls': '-',  'lw': 2.5, 'ms': 8,
           'label': '暴露度 E'},
    'S': {'color': '#F39C12', 'marker': 's', 'ls': '--', 'lw': 2.5, 'ms': 8,
           'label': '敏感性 S'},
    'C': {'color': '#2ECC71', 'marker': '^', 'ls': ':',  'lw': 2.5, 'ms': 8,
           'label': '应对不足 C'},
}

for key, mean, std, c in zip(['E','S','C'],
                               [E_m, S_m, C_m],
                               [E_s, S_s, C_s],
                               [cfg['E'], cfg['S'], cfg['C']]):
    # 阴影（均值±0.5*Std，不使用整个Std避免遮挡）
    ax.fill_between(years, mean - 0.5*std, mean + 0.5*std,
                     alpha=0.12, color=c['color'])
    # 主线
    ax.plot(years, mean,
             color=c['color'], marker=c['marker'],
             linestyle=c['ls'], linewidth=c['lw'],
             markersize=c['ms'], label=c['label'], zorder=5)
    # 末尾数值标注
    ax.annotate(f"{mean[-1]:.4f}",
                xy=(years[-1], mean[-1]),
                xytext=(8, 0), textcoords='offset points',
                fontsize=9.5, color=c['color'], va='center', fontweight='bold')

# 线性趋势线（仅主线延伸）
for key, mean, c in zip(['E','S','C'], [E_m, S_m, C_m],
                          [cfg['E'], cfg['S'], cfg['C']]):
    z = np.polyfit(years, mean, 1)
    p = np.poly1d(z)
    direction = '↑' if z[0] > 0 else '↓'
    ax.plot(years, p(years), color=c['color'], linewidth=1.2,
             linestyle='-', alpha=0.4)
    # 斜率标注（右侧）
    ax.annotate(f"趋势: {z[0]:+.5f}/年 {direction}",
                xy=(years[len(years)//2], p(years[len(years)//2])),
                fontsize=8, color=c['color'], alpha=0.8,
                xytext=(0, 10 if key=='E' else (-15 if key=='S' else -30)),
                textcoords='offset points')

ax.set_xlabel('年份', fontsize=13)
ax.set_ylabel('分量均值（归一化）', fontsize=13)
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], rotation=30, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.legend(fontsize=11, frameon=True, loc='upper left',
           framealpha=0.9, edgecolor='#CCCCCC')
ax.grid(axis='y', alpha=0.35, linestyle='--')
ax.grid(axis='x', alpha=0.15)

# 添加年际降雨量参考（副轴，仅均值折线）
rain_dyn_means = []
for yr in STUDY_YEARS:
    r = year_dyn[yr]['rain']
    if r is not None:
        v = r[valid_mask & np.isfinite(r)]
        rain_dyn_means.append(float(v.mean()))
    else:
        rain_dyn_means.append(np.nan)

ax2 = ax.twinx()
ax2.bar(years, rain_dyn_means, color='#AED6F1', alpha=0.35,
         width=0.6, label='汛期降雨量（mm）', zorder=1)
ax2.set_ylabel('汛期累积降雨量 (mm)', fontsize=11, color='#2980B9')
ax2.tick_params(axis='y', labelcolor='#2980B9')
ax2.set_ylim(0, max([r for r in rain_dyn_means if not np.isnan(r)]) * 2.0)
ax2.legend(loc='upper right', fontsize=10)

plt.title('图9 北京市内涝脆弱性三因子分量时序演变（2012-2024）\n'
           '（折线±阴影为均值±0.5Std；柱状为汛期降雨量参考）',
           fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()

out_fig = os.path.join(OUTPUT_DIR, 'ESC_Annual_Trend_Fig9.png')
fig.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  ESC_Annual_Trend_Fig9.png saved")

# 同时保存论文表格格式的Excel（方便直接插入论文）
table16_data = {
    '年份':      [int(y) for y in df_esc['年份']],
    'E均值':     df_esc['E均值'].round(4).tolist(),
    'E标准差':   df_esc['E标准差'].round(4).tolist(),
    'S均值':     df_esc['S均值'].round(4).tolist(),
    'S标准差':   df_esc['S标准差'].round(4).tolist(),
    'C均值':     df_esc['C均值'].round(4).tolist(),
    'C标准差':   df_esc['C标准差'].round(4).tolist(),
}
df_table16 = pd.DataFrame(table16_data)
out_table = os.path.join(RISK_DIR, 'Table16_ESC_Annual.csv')
df_table16.to_csv(out_table, index=False, encoding='utf-8-sig')
print(f"  Table16_ESC_Annual.csv saved（可直接插入论文表格）")

print("\n完成！")
print(f"  {out_csv}")
print(f"  {out_table}")
print(f"  {out_fig}")