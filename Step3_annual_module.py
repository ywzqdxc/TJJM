"""
Step 3 (v7.0): VSC三维洪涝脆弱性评估 + 逐年脆弱性计算
=========================================================
v7.0 相对 v6.0 的改动：
  在原有多年综合评估结束后，新增逐年脆弱性计算模块（★ 标记处）：
  - 静态指标（TWI/HAND/WP/PD/RD/SH/HO/FS）保持不变
  - 动态指标（降雨量R/径流CR）按年替换
  - 全局归一化边界固定（基于所有年份的全局min/max），保证年际可比
  - 使用同一套 w_combo 权重计算TOPSIS得分
  - 输出 Vuln_TOPSIS_2012.npy ... Vuln_TOPSIS_2024.npy
  - 保存 Annual_Vuln_Stats.csv 和 Global_Norm_Bounds.npy

其余逻辑与 v6.0 完全一致（不重复粘贴，以import方式复用）。
若需要完整代码，将以下模块追加到 step3_v6.py 末尾即可。
"""

# ============================================================
# 以下代码追加到 step3_v6.py 末尾
# （在"汇总"print语句之前插入）
# ============================================================

# ★★★ 追加起始位置：step3_v6.py 最后一个 print 语句之前 ★★★

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import os, warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径（与 step3_v6.py 中的变量保持一致）
# ============================================================
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
STATIC_DIR = r'./Step_New/Static'
DYN_DIR    = r'./Step_New/Dynamic'
EXT_DIR    = r'./Step_New/External'
OUTPUT_DIR = r'./Step_New/Risk_Map'
STUDY_YEARS = list(range(2012, 2025))   # 13年

print("\n" + "=" * 70)
print("★ v7.0 新增模块：逐年脆弱性计算")
print("=" * 70)

# ============================================================
# 工具函数（与step3_v6.py中一致，此处重定义保证独立运行）
# ============================================================

def _load_npy(path):
    if not os.path.exists(path): return None
    return np.load(path).astype(np.float32)

def _load_tif(path, h, w, dst_crs, dst_transform):
    if not os.path.exists(path): return None
    with rasterio.open(path) as src:
        if src.height == h and src.width == w and src.crs == dst_crs:
            arr = src.read(1).astype(np.float32)
        else:
            arr = np.full((h, w), -9999.0, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=arr,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, dst_nodata=-9999.0)
        nd = src.nodata
        if nd is not None: arr = np.where(arr == nd, np.nan, arr)
    return np.where(arr < -1e10, np.nan, arr)


# ============================================================
# 重新加载基准信息
# ============================================================
with rasterio.open(DEM_PATH) as ref:
    h_g, w_g       = ref.height, ref.width
    dst_crs_g       = ref.crs
    dst_transform_g = ref.transform
    out_profile_g   = ref.profile.copy()
out_profile_g.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='deflate')

nodata_mask_g = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask_g  = ~nodata_mask_g

# 重新加载 w_combo（从保存的权重CSV）
weights_csv = os.path.join(OUTPUT_DIR, 'VSC_Weights.csv')
if os.path.exists(weights_csv):
    df_w_saved = pd.read_csv(weights_csv)
    w_combo_g  = df_w_saved['组合'].values.astype(np.float64)
    print(f"  ✅ 从 VSC_Weights.csv 加载组合权重: {[f'{v:.4f}' for v in w_combo_g]}")
else:
    print(f"  ❌ VSC_Weights.csv 不存在，请先运行 step3_v6.py 的主体部分")
    raise FileNotFoundError("VSC_Weights.csv not found")

# ============================================================
# 加载8个静态指标（不随年份变化）
# ============================================================
print("\n[1/4] 加载静态指标...")

static_raw = {
    'twi':  _load_npy(os.path.join(STATIC_DIR, 'twi.npy')),     # E
    'hand': _load_npy(os.path.join(STATIC_DIR, 'hand.npy')),    # E（逆向）
    'wp':   _load_tif(os.path.join(EXT_DIR, 'waterlogging_point_density_30m.tif'),
                      h_g, w_g, dst_crs_g, dst_transform_g),    # E
    'pop':  _load_tif(os.path.join(EXT_DIR, 'population_density_30m.tif'),
                      h_g, w_g, dst_crs_g, dst_transform_g),    # S
    'road': _load_tif(os.path.join(EXT_DIR, 'road_density_30m.tif'),
                      h_g, w_g, dst_crs_g, dst_transform_g),    # S
    'sh':   _load_tif(os.path.join(EXT_DIR, 'shelter_density_30m.tif'),
                      h_g, w_g, dst_crs_g, dst_transform_g),    # C（逆向）
    'ho':   _load_tif(os.path.join(EXT_DIR, 'hospital_density_30m.tif'),
                      h_g, w_g, dst_crs_g, dst_transform_g),    # C（逆向）
    'fs':   _load_tif(os.path.join(EXT_DIR, 'firestation_density_30m.tif'),
                      h_g, w_g, dst_crs_g, dst_transform_g),    # C（逆向）
}

for k, v in static_raw.items():
    if v is None:
        print(f"  ❌ {k} 静态指标缺失！")
    else:
        valid_v = v[valid_mask_g & np.isfinite(v)]
        print(f"  ✅ {k:6s}: 有效={valid_v.size:,}  均值={valid_v.mean():.4f}")


# ============================================================
# 计算全局归一化边界（基于所有年份，保证年际可比）
# ============================================================
print("\n[2/4] 计算全局归一化边界（所有年份 + 静态指标）...")

# 收集所有年份的降雨量和CR
all_rain_vals = []
all_cr_vals   = []
year_dyn_data = {}   # {year: {'rain': arr, 'cr': arr}}

rain_mean_fallback = _load_npy(os.path.join(DYN_DIR, 'Precipitation_Mean.npy'))
cr_mean_fallback   = _load_npy(os.path.join(DYN_DIR, 'CR_MultiYear_Mean.npy'))

for year in STUDY_YEARS:
    rain_yr = _load_npy(os.path.join(DYN_DIR, f'Precipitation_{year}.npy'))
    cr_yr   = _load_npy(os.path.join(DYN_DIR, f'CR_{year}.npy'))

    if rain_yr is None:
        rain_yr = rain_mean_fallback
        if year == STUDY_YEARS[0]:
            print(f"  ⚠️  无逐年降雨文件，使用多年均值代替（请运行step2_v7.py）")
    if cr_yr is None:
        cr_yr = cr_mean_fallback

    year_dyn_data[year] = {'rain': rain_yr, 'cr': cr_yr}

    if rain_yr is not None:
        all_rain_vals.append(rain_yr[valid_mask_g & np.isfinite(rain_yr)])
    if cr_yr is not None:
        all_cr_vals.append(cr_yr[valid_mask_g & np.isfinite(cr_yr)])

# 全局边界
rain_all = np.concatenate(all_rain_vals) if all_rain_vals else np.array([0, 800])
cr_all   = np.concatenate(all_cr_vals)   if all_cr_vals   else np.array([0.05, 0.95])

GLOBAL_BOUNDS = {
    'rain_min': float(np.percentile(rain_all, 0.5)),
    'rain_max': float(np.percentile(rain_all, 99.5)),
    'cr_min':   float(cr_all.min()),
    'cr_max':   float(cr_all.max()),
}
print(f"  降雨量全局边界: [{GLOBAL_BOUNDS['rain_min']:.1f}, {GLOBAL_BOUNDS['rain_max']:.1f}] mm")
print(f"  CR全局边界:     [{GLOBAL_BOUNDS['cr_min']:.4f}, {GLOBAL_BOUNDS['cr_max']:.4f}]")

# 静态指标的边界（与多年综合计算一致：基于valid_mask内有效像元）
def _get_bounds(arr, valid_mask, plo=0.5, phi=99.5):
    vals = arr[valid_mask & np.isfinite(arr)]
    if vals.size < 10: return (0.0, 1.0)
    return (float(np.percentile(vals, plo)), float(np.percentile(vals, phi)))

# 计算静态指标的全局边界
static_bounds = {}
for k, arr in static_raw.items():
    if arr is not None:
        static_bounds[k] = _get_bounds(arr, valid_mask_g)

# 保存全局边界（供Step6复用）
bounds_dict = {**GLOBAL_BOUNDS, **{f'{k}_min':v[0] for k,v in static_bounds.items()},
               **{f'{k}_max':v[1] for k,v in static_bounds.items()}}
np.save(os.path.join(OUTPUT_DIR, 'Global_Norm_Bounds.npy'), np.array([0]))  # 占位
pd.DataFrame([bounds_dict]).to_csv(
    os.path.join(OUTPUT_DIR, 'Global_Norm_Bounds.csv'), index=False)
print(f"  ✅ Global_Norm_Bounds.csv 已保存")


# ============================================================
# 归一化函数（固定边界版）
# ============================================================
def minmax_fixed(arr, lo, hi, valid_mask, direction='positive'):
    """使用固定边界归一化，保证年际可比"""
    result = np.full_like(arr, np.nan, dtype=np.float32)
    if hi - lo < 1e-10:
        result[valid_mask & np.isfinite(arr)] = 0.5
        return result
    normed = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if direction == 'negative': normed = 1.0 - normed
    result[valid_mask & np.isfinite(arr)] = normed[valid_mask & np.isfinite(arr)]
    return result


def build_year_xflat(rain_yr, cr_yr, valid_px_mask):
    """
    构建单年的像元矩阵 X_flat (n_pixels, 10)
    指标顺序：[R, CR, TWI, HAND(逆), WP, PD, RD, SH(逆), HO(逆), FS(逆)]
    """
    # 动态指标归一化（固定全局边界）
    r_n  = minmax_fixed(rain_yr, GLOBAL_BOUNDS['rain_min'],
                         GLOBAL_BOUNDS['rain_max'], valid_mask_g, 'positive')
    cr_n = minmax_fixed(cr_yr,   GLOBAL_BOUNDS['cr_min'],
                         GLOBAL_BOUNDS['cr_max'],   valid_mask_g, 'positive')

    # 静态指标归一化（固定边界）
    def sn(key, dire):
        arr = static_raw[key]
        if arr is None: return np.where(valid_mask_g, 0.0, np.nan).astype(np.float32)
        lo, hi = static_bounds.get(key, (0.0, 1.0))
        return minmax_fixed(arr, lo, hi, valid_mask_g, dire)

    twi_n  = sn('twi',  'positive')
    hand_n = sn('hand', 'negative')
    wp_n   = sn('wp',   'positive')
    pop_n  = sn('pop',  'positive')
    road_n = sn('road', 'positive')
    sh_n   = sn('sh',   'negative')
    ho_n   = sn('ho',   'negative')
    fs_n   = sn('fs',   'negative')

    norm_list = [r_n, cr_n, twi_n, hand_n, wp_n,
                 pop_n, road_n, sh_n, ho_n, fs_n]

    # 构建有效像元掩膜
    px_mask = valid_mask_g.copy()
    for arr in norm_list:
        px_mask &= np.isfinite(arr)

    X = np.column_stack([arr[px_mask] for arr in norm_list]).astype(np.float64)
    X = np.clip(X, 0.0, 1.0)
    return X, px_mask


def topsis_score(X, w):
    """TOPSIS评分（与step3_v6.py中完全一致）"""
    Xw    = X * w[np.newaxis, :]
    Z_pos = Xw.max(axis=0); Z_neg = Xw.min(axis=0)
    D_pos = np.sqrt(((Xw - Z_pos)**2).sum(axis=1))
    D_neg = np.sqrt(((Xw - Z_neg)**2).sum(axis=1))
    return (D_neg / (D_pos + D_neg + 1e-10)).astype(np.float32)


# ============================================================
# 逐年计算TOPSIS脆弱性得分
# ============================================================
print("\n[3/4] 逐年计算TOPSIS脆弱性得分...")

annual_stats = []
year_vuln_means = []
year_vuln_stds  = []

for year in STUDY_YEARS:
    dyn  = year_dyn_data.get(year, {})
    rain = dyn.get('rain', rain_mean_fallback)
    cr   = dyn.get('cr',   cr_mean_fallback)

    if rain is None or cr is None:
        print(f"  {year}: ⚠️  动态数据缺失，跳过")
        year_vuln_means.append(np.nan)
        year_vuln_stds.append(np.nan)
        continue

    X_yr, px_mask_yr = build_year_xflat(rain, cr, valid_mask_g)
    s_yr = topsis_score(X_yr, w_combo_g)

    # 还原到空间矩阵
    vuln_yr = np.full((h_g, w_g), np.nan, dtype=np.float32)
    vuln_yr[px_mask_yr] = s_yr

    # 保存NPY
    out_npy = os.path.join(OUTPUT_DIR, f'Vuln_TOPSIS_{year}.npy')
    np.save(out_npy, vuln_yr)

    # 统计
    v_valid = s_yr
    mean_v  = float(v_valid.mean())
    std_v   = float(v_valid.std())
    p5_v    = float(np.percentile(v_valid, 5))
    p25_v   = float(np.percentile(v_valid, 25))
    p50_v   = float(np.percentile(v_valid, 50))
    p75_v   = float(np.percentile(v_valid, 75))
    p95_v   = float(np.percentile(v_valid, 95))

    year_vuln_means.append(mean_v)
    year_vuln_stds.append(std_v)

    annual_stats.append({
        '年份': year, '有效像元': len(v_valid),
        '均值': mean_v, '标准差': std_v,
        'P5': p5_v, 'P25': p25_v, 'P50': p50_v, 'P75': p75_v, 'P95': p95_v
    })

    # 判断数据是否使用了逐年文件
    yr_file_exists = os.path.exists(os.path.join(DYN_DIR, f'Precipitation_{year}.npy'))
    flag = '(逐年)' if yr_file_exists else '(均值代替)'
    print(f"  {year} {flag}: 均值={mean_v:.4f}  Std={std_v:.4f}  "
          f"P5={p5_v:.4f}  P95={p95_v:.4f}")

# 保存年度统计CSV
df_annual = pd.DataFrame(annual_stats)
df_annual.to_csv(os.path.join(OUTPUT_DIR, 'Annual_Vuln_Stats.csv'),
                  index=False, encoding='utf-8-sig')
print(f"\n  ✅ Annual_Vuln_Stats.csv 已保存")
print(f"  ✅ Vuln_TOPSIS_{{year}}.npy × {len(annual_stats)} 个年份")


# ============================================================
# 逐年演变可视化（年均值折线图 + 标准差阴影）
# ============================================================
print("\n[4/4] 绘制逐年脆弱性演变图...")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

VIS_DIR = r'./Step_New/Visualization/Step3_VSC'
os.makedirs(VIS_DIR, exist_ok=True)

years_arr = np.array([s['年份'] for s in annual_stats])
means_arr = np.array([s['均值'] for s in annual_stats])
stds_arr  = np.array([s['标准差'] for s in annual_stats])
p5_arr    = np.array([s['P5'] for s in annual_stats])
p95_arr   = np.array([s['P95'] for s in annual_stats])

# 从逐年降雨量文件获取全域均值（用于右轴）
rain_ts_list = []
for year in STUDY_YEARS:
    fp = os.path.join(DYN_DIR, f'Precipitation_{year}.npy')
    if os.path.exists(fp):
        arr = np.load(fp)
        v   = arr[valid_mask_g & np.isfinite(arr)]
        rain_ts_list.append(float(v.mean()) if len(v) > 0 else np.nan)
    else:
        rain_ts_list.append(np.nan)
rain_ts_arr = np.array(rain_ts_list)

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.35)

# ── 上图：脆弱性时序（折线+置信区间）
ax1 = fig.add_subplot(gs[0])
ax1.fill_between(years_arr, means_arr - stds_arr, means_arr + stds_arr,
                  alpha=0.2, color='#D7191C', label='均值±标准差')
ax1.fill_between(years_arr, p5_arr, p95_arr,
                  alpha=0.10, color='#FDAE61', label='P5-P95区间')
ax1.plot(years_arr, means_arr, color='#D7191C', marker='o',
          linewidth=2.5, markersize=8, label='全域风险均值', zorder=5)
ax1.plot(years_arr, p95_arr, color='#E74C3C', linewidth=1.2,
          linestyle=':', marker='v', markersize=5, label='P95极值')

# 添加线性趋势线
valid_y = ~np.isnan(means_arr)
if valid_y.sum() >= 3:
    z   = np.polyfit(years_arr[valid_y], means_arr[valid_y], 1)
    p   = np.poly1d(z)
    ax1.plot(years_arr, p(years_arr), 'k--', linewidth=1.5, alpha=0.6,
              label=f'线性趋势 ({z[0]:+.5f}/年)')

ax1.set_ylabel('TOPSIS综合风险指数', fontsize=12, fontweight='bold')
ax1.set_xlim(STUDY_YEARS[0] - 0.5, STUDY_YEARS[-1] + 0.5)
ax1.set_ylim(0, 1.0)
ax1.set_xticks(years_arr)
ax1.set_xticklabels([str(y) for y in STUDY_YEARS], rotation=30, fontsize=9)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_title('北京市城市内涝综合风险逐年演变（2012-2024）\n'
               '（静态指标固定，动态指标：降雨量R、径流系数CR逐年更新）',
               fontsize=13, fontweight='bold', pad=10)

# ── 下图：降雨量柱状图
ax2 = fig.add_subplot(gs[1])
valid_rain = ~np.isnan(rain_ts_arr)
if valid_rain.sum() > 0:
    ax2.bar(years_arr[valid_rain], rain_ts_arr[valid_rain],
             color='#74a9cf', alpha=0.8, width=0.6, label='汛期降雨量均值')
    ax2.set_ylabel('降雨量 (mm)', fontsize=11)
    ax2.set_xlim(STUDY_YEARS[0] - 0.5, STUDY_YEARS[-1] + 0.5)
    ax2.set_xticks(years_arr)
    ax2.set_xticklabels([str(y) for y in STUDY_YEARS], rotation=30, fontsize=9)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

plt.savefig(os.path.join(VIS_DIR, 'VSC_Annual_Trend.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ VSC_Annual_Trend.png")

print("\n" + "=" * 70)
print("Step 3 v7.0 逐年计算模块完成！")
print(f"  逐年NPY: Vuln_TOPSIS_2012.npy ... Vuln_TOPSIS_2024.npy")
print(f"  统计CSV: Annual_Vuln_Stats.csv")
print(f"  边界CSV: Global_Norm_Bounds.csv")
print(f"  演变图:  VSC_Annual_Trend.png")
print(f"\n  ⬇️  下一步：运行 step6_mk_trend.py（MK趋势检验）")
print("=" * 70)