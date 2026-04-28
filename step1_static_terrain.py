"""
Step 1: 静态下垫面预处理 (融合高精度土壤水文参数)
=====================================
提取流向(fdir)、坡度(Slope)、HAND。
处理 Ks、Theta_s、Psi，识别城市不透水层，生成物理下垫面。

修订说明 (v2.1):
  - 土壤参数路径更新为裁剪+重采样后的三个独立TIF文件
  - 新增 Psi (饱和毛管势) 参数处理
  - 修正 nodata 识别: float32 nodata=-3.4e+38 需用阈值判断
  - 修正城市识别逻辑: θs 有效值范围 0.40~0.60，原 >1 判断失效
    改为: nodata区域 → 城市/不透水层（Ks=0.01, θs=0.10, Psi=-3.0 cm）
  - Psi 均为负值(cm)，正常物理范围，直接使用绝对值作为毛管吸力
  - 三个TIF与DEM有小偏差（5850×7560 vs 5821×7530），使用 out_shape 重采样对齐
"""

import numpy as np
import rasterio
from rasterio.plot import show
from pysheds.grid import Grid
from scipy.ndimage import sobel
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
DEM_PATH   = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'

# 裁剪+重采样后的土壤参数 TIF（5850×7560，与DEM略有偏差，读取时对齐）
KS_PATH    = r'F:\Data\src\fac\KSCH\ksch\Extract_K_SC1_Resample1.tif'
THS_PATH   = r'F:\Data\src\fac\THSCH\thsch\Extract_THSC1_Resample1.tif'
PSI_PATH   = r'F:\Data\src\fac\PSI\psi\Extract_PSI_1_Resample1.tif'

OUTPUT_DIR = r'./Step_New/Static'
VIS_DIR    = r'./Step_New/Visualization/Step1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,    exist_ok=True)

# ============================================================
# 工具函数
# ============================================================
# float32 nodata 阈值（-3.4e+38 的安全判断值）
NODATA_THRESHOLD = -1e30

def read_tif_aligned(path, h, w, nodata_threshold=NODATA_THRESHOLD):
    """
    读取TIF并强制对齐到 (h, w)，处理 nodata。
    返回: data_clean (float32, nodata→nan), valid_mask (bool, True=有效)
    """
    with rasterio.open(path) as src:
        raw = src.read(
            1,
            out_shape=(h, w),
            resampling=rasterio.enums.Resampling.bilinear
        ).astype(np.float32)
        nodata_val = src.nodata

    # 构造 nodata mask
    invalid = np.zeros(raw.shape, dtype=bool)
    if nodata_val is not None:
        # float32 极大负数的安全比较
        if nodata_val < nodata_threshold:
            invalid |= (raw < nodata_threshold)
        else:
            invalid |= (raw == nodata_val)
    else:
        # 自动检测极大负值
        invalid |= (raw < nodata_threshold)
    invalid |= ~np.isfinite(raw)

    data_clean = np.where(invalid, np.nan, raw)
    return data_clean, ~invalid


def safe_percentile(arr, pct):
    return float(np.nanpercentile(arr, pct))


def print_stats(name, arr):
    valid = arr[~np.isnan(arr)]
    stats_dict = {
        'name':  name,
        'count': int(valid.size),
        'min':   float(np.min(valid)),
        'max':   float(np.max(valid)),
        'mean':  float(np.mean(valid)),
        'std':   float(np.std(valid)),
        'cv':    float(np.std(valid) / (abs(np.mean(valid)) + 1e-10)),
        'p25':   float(np.percentile(valid, 25)),
        'p50':   float(np.percentile(valid, 50)),
        'p75':   float(np.percentile(valid, 75)),
        'skew':  float(stats.skew(valid)),
        'kurt':  float(stats.kurtosis(valid)),
    }
    print(f"\n  [{name}]")
    print(f"    有效像元: {stats_dict['count']:,}")
    print(f"    Min={stats_dict['min']:.4f}  Max={stats_dict['max']:.4f}")
    print(f"    Mean={stats_dict['mean']:.4f}  Std={stats_dict['std']:.4f}  CV={stats_dict['cv']:.4f}")
    print(f"    P25={stats_dict['p25']:.4f}  P50={stats_dict['p50']:.4f}  P75={stats_dict['p75']:.4f}")
    print(f"    偏度={stats_dict['skew']:.4f}  峰度={stats_dict['kurt']:.4f}")
    return stats_dict


def clip_for_display(arr, lo=1, hi=99):
    valid = arr[~np.isnan(arr)]
    lo_v  = np.percentile(valid, lo)
    hi_v  = np.percentile(valid, hi)
    return np.clip(arr, lo_v, hi_v)


# ============================================================
# 1/4  初始化 DEM 与 pysheds
# ============================================================
print("=" * 70)
print("Step 1：静态下垫面预处理")
print("=" * 70)
print("\n[1/4] 初始化 DEM 与 Pysheds 物理网格...")

grid = Grid.from_raster(DEM_PATH)
dem  = grid.read_raster(DEM_PATH)

with rasterio.open(DEM_PATH) as ref:
    profile   = ref.profile.copy()
    h, w      = ref.height, ref.width
    transform = ref.transform
    crs       = ref.crs

dem_clean   = np.where(dem < -100, np.nan, dem.astype(np.float32))
nodata_mask = np.isnan(dem_clean)          # True = DEM无效（水体/边界外）
valid_pixels = int(np.sum(~nodata_mask))

print(f"    栅格尺寸: {h} 行 × {w} 列")
print(f"    有效像元: {valid_pixels:,}  /  总像元: {h * w:,}")

dem_stats = print_stats('DEM高程(m)', dem_clean)

# ============================================================
# 2/4  计算基础地形水文特征
# ============================================================
print("\n[2/4] 计算基础地形水文特征...")

# --- 坡度 ---
dy = sobel(dem_clean, axis=0) / 30.0
dx = sobel(dem_clean, axis=1) / 30.0
slope_arr = np.clip(np.rad2deg(np.arctan(np.sqrt(dx**2 + dy**2))), 0.01, 60.0)
slope_arr = np.where(nodata_mask, np.nan, slope_arr)

# --- 流向 ---
pit_filled = grid.fill_pits(dem)
flooded    = grid.fill_depressions(pit_filled)
inflated   = grid.resolve_flats(flooded)
dirmap     = (64, 128, 1, 2, 4, 8, 16, 32)
fdir       = grid.flowdir(inflated, dirmap=dirmap)

# --- 汇流累积量 ---
acc          = grid.accumulation(fdir, dirmap=dirmap)
channel_mask = acc > 500

# --- HAND ---
try:
    hand     = grid.compute_hand(fdir, inflated, channel_mask, dirmap=dirmap)
    hand_arr = np.clip(np.array(hand, dtype=np.float32), 0, 500)
    print("    HAND 计算成功（pysheds 原生）")
except Exception as e:
    print(f"    HAND 计算回退至归一化DEM代理 ({e})")
    dem_norm = (dem_clean - np.nanmin(dem_clean)) / (np.nanmax(dem_clean) - np.nanmin(dem_clean) + 1e-8)
    hand_arr = dem_norm * 100.0

hand_arr  = np.where(nodata_mask, np.nan, hand_arr)
fdir_arr  = np.array(fdir, dtype=np.float32)
acc_arr   = np.array(acc,  dtype=np.float32)
acc_arr   = np.where(nodata_mask, np.nan, acc_arr)

slope_stats = print_stats('坡度(°)', slope_arr)
hand_stats  = print_stats('HAND(m)', hand_arr)

# ============================================================
# 3/4  读取土壤物理参数 (Ks, θs, Psi) 并识别城市不透水层
# ============================================================
print("\n[3/4] 解析土壤物理参数 (Ks, θs, Psi) 并识别城市硬化面...")

# --- 读取三个参数 ---
ks_raw,  ks_valid_mask  = read_tif_aligned(KS_PATH,  h, w)
ths_raw, ths_valid_mask = read_tif_aligned(THS_PATH, h, w)
psi_raw, psi_valid_mask = read_tif_aligned(PSI_PATH, h, w)

# --- 诊断原始值范围 ---
print(f"    原始 Ks  有效值范围: [{np.nanmin(ks_raw):.4f}, {np.nanmax(ks_raw):.4f}] mm/h")
print(f"    原始 θs  有效值范围: [{np.nanmin(ths_raw):.4f}, {np.nanmax(ths_raw):.4f}] m³/m³")
print(f"    原始 Psi 有效值范围: [{np.nanmin(psi_raw):.4f}, {np.nanmax(psi_raw):.4f}] cm")

# ---------------------------------------------------------------
# 城市不透水层识别策略：
#   土壤参数 TIF 的 nodata 区（~ks_valid_mask）且 DEM 有效区
#   = 城市建成区/硬化地面（无自然土壤数据）
#   补充：Ks 极低（< 1 mm/h）也视为不透水
# ---------------------------------------------------------------
soil_nodata  = ~ks_valid_mask | ~ths_valid_mask  # 任一土壤参数缺失
urban_mask   = (soil_nodata & ~nodata_mask)       # DEM 有效但土壤缺失 → 城市

# 额外：极低 Ks 的已识别自然区也标为半透水（保留原值，不强制覆盖）
# 这里城市区强制赋物理合理的不透水值
KS_URBAN   = 0.01    # mm/h  (几乎不透水)
THS_URBAN  = 0.10    # m³/m³ (极低含水量)
PSI_URBAN  = -3.00   # cm    (极小毛管吸力)

# --- Ks ---
ks = ks_raw.copy()
ks[urban_mask]  = KS_URBAN
ks = np.clip(ks, 0.01, 200.0)
ks = np.where(nodata_mask, np.nan, ks)

# --- θs ---
ths = ths_raw.copy()
ths[urban_mask] = THS_URBAN
ths = np.clip(ths, 0.05, 0.65)
ths = np.where(nodata_mask, np.nan, ths)

# --- Psi (毛管吸力，原始值为负，物理上取绝对值，单位 cm)
#   psi_raw 范围约 -54 ~ -6 cm，负号表示吸力方向（Green-Ampt 中用 |ψ|）
psi_abs = np.abs(psi_raw)           # 转换为正值吸力 (cm)
psi_abs[urban_mask] = abs(PSI_URBAN)
psi_abs = np.clip(psi_abs, 0.1, 100.0)
psi_abs = np.where(nodata_mask, np.nan, psi_abs)

urban_pct = float(np.sum(urban_mask)) / valid_pixels * 100
print(f"    识别城市不透水层像元: {int(np.sum(urban_mask)):,}  ({urban_pct:.2f}% 全域面积)")
print(f"    → 城市区赋值: Ks={KS_URBAN} mm/h, θs={THS_URBAN}, |Psi|={abs(PSI_URBAN)} cm")

ks_stats  = print_stats('饱和导水率Ks(mm/h)',  ks)
ths_stats = print_stats('饱和含水量θs',        ths)
psi_stats = print_stats('毛管吸力|Psi|(cm)',   psi_abs)

# ============================================================
# 保存 NPY 特征文件
# ============================================================
np.save(os.path.join(OUTPUT_DIR, 'fdir.npy'),        fdir_arr)
np.save(os.path.join(OUTPUT_DIR, 'slope.npy'),       slope_arr)
np.save(os.path.join(OUTPUT_DIR, 'hand.npy'),        hand_arr)
np.save(os.path.join(OUTPUT_DIR, 'acc.npy'),         acc_arr)
np.save(os.path.join(OUTPUT_DIR, 'nodata_mask.npy'), nodata_mask)
np.save(os.path.join(OUTPUT_DIR, 'ks.npy'),          ks)
np.save(os.path.join(OUTPUT_DIR, 'ths.npy'),         ths)
np.save(os.path.join(OUTPUT_DIR, 'psi.npy'),         psi_abs)
np.save(os.path.join(OUTPUT_DIR, 'urban_mask.npy'),  urban_mask)
print("\n    所有特征已保存至 NPY（新增 psi.npy）。")

# ============================================================
# 4/4  完整可视化
# ============================================================
print("\n[4/4] 生成可视化图像...")

# ----------------------------------------------------------
# 图1：七图总览（新增 Psi）
# ----------------------------------------------------------
fig = plt.figure(figsize=(24, 14))
fig.suptitle('Step 1：北京市静态下垫面特征提取总览\n(基于30m NASA DEM + 土壤水文参数 Ks/θs/Psi)',
             fontsize=15, fontweight='bold', y=0.99)

gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30)

panels = [
    (fig.add_subplot(gs[0, 0]), dem_clean,  'terrain',    '① DEM 高程 (m)',            '高程 (m)',    dem_stats),
    (fig.add_subplot(gs[0, 1]), slope_arr,  'RdYlGn_r',   '② 坡度 (°)',                '坡度 (°)',    slope_stats),
    (fig.add_subplot(gs[0, 2]), hand_arr,   'Blues_r',    '③ HAND (m)',                'HAND (m)',    hand_stats),
    (fig.add_subplot(gs[0, 3]), ks,         'YlOrRd',     '④ 饱和导水率 Ks (mm/h)',    'Ks (mm/h)',   ks_stats),
    (fig.add_subplot(gs[1, 0]), ths,        'BrBG',       '⑤ 饱和含水量 θs',          'θs (m³/m³)', ths_stats),
    (fig.add_subplot(gs[1, 1]), psi_abs,    'PuBu',       '⑥ 毛管吸力 |Psi| (cm)',    '|Psi| (cm)', psi_stats),
]

for ax, data, cmap, title, cb_label, st in panels:
    disp = clip_for_display(data, hi=98)
    im   = ax.imshow(disp, cmap=cmap)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label(cb_label, fontsize=8)
    ax.text(0.02, 0.02,
            f"均={st['mean']:.3f}\nCV={st['cv']:.3f}",
            transform=ax.transAxes, fontsize=8.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

# 第7图：θs 叠加城市掩膜
ax6 = fig.add_subplot(gs[1, 2])
ths_disp = clip_for_display(ths)
im6 = ax6.imshow(ths_disp, cmap='BrBG')
urban_overlay = np.where(urban_mask & ~nodata_mask, 1.0, np.nan)
ax6.imshow(urban_overlay, cmap='Reds', alpha=0.65, vmin=0, vmax=1)
ax6.set_title('⑦ θs 叠加城市硬化（红色）', fontsize=10, fontweight='bold')
ax6.axis('off')
cb6 = plt.colorbar(im6, ax=ax6, shrink=0.82)
cb6.set_label('θs (m³/m³)', fontsize=8)
ax6.text(0.02, 0.02,
         f"城市占比={urban_pct:.1f}%",
         transform=ax6.transAxes, fontsize=8.5, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

# 第8图：汇流累积量
ax7 = fig.add_subplot(gs[1, 3])
acc_log = np.log1p(acc_arr)
acc_log = np.where(nodata_mask, np.nan, acc_log)
im7 = ax7.imshow(acc_log, cmap='Blues')
river_mask = (acc_arr > 500) & ~nodata_mask
ax7.imshow(np.where(river_mask, 1.0, np.nan), cmap='Reds', alpha=0.75, vmin=0, vmax=1)
ax7.set_title('⑧ 汇流量(对数)+水系（红色）', fontsize=10, fontweight='bold')
ax7.axis('off')
plt.colorbar(im7, ax=ax7, shrink=0.82, label='log(acc+1)')

out1 = os.path.join(VIS_DIR, 'Step1_01_Overview.png')
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"    [图1] 八图总览  → {out1}")

# ----------------------------------------------------------
# 图2：统计分布直方图（5个关键指标）
# ----------------------------------------------------------
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.suptitle('Step 1：关键地形与土壤特征统计分布', fontsize=14, fontweight='bold')

hist_configs = [
    (slope_arr, '坡度 (°)',       'steelblue',   slope_stats),
    (hand_arr,  'HAND (m)',       'seagreen',    hand_stats),
    (ks,        'Ks (mm/h)',      'darkorange',  ks_stats),
    (ths,       'θs (m³/m³)',    'saddlebrown', ths_stats),
    (psi_abs,   '|Psi| (cm)',     'mediumpurple',psi_stats),
]
for i, (ax, (data, label, color, st)) in enumerate(zip(axes2.flat, hist_configs)):
    valid_data = data[~np.isnan(data)]
    upper = np.percentile(valid_data, 99.5)
    d_trim = valid_data[valid_data <= upper]
    ax.hist(d_trim, bins=80, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(st['mean'], color='red',    linestyle='--', linewidth=1.5, label=f"均值={st['mean']:.4f}")
    ax.axvline(st['p50'],  color='orange', linestyle='-',  linewidth=1.5, label=f"中位={st['p50']:.4f}")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('像元频数', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.text(0.97, 0.97,
            f"N={st['count']:,}\nStd={st['std']:.4f}\nCV={st['cv']:.4f}\n偏度={st['skew']:.3f}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 第6图：城市/自然区饼图
ax_pie = axes2.flat[5]
sizes  = [int((~urban_mask & ~nodata_mask).sum()), int(urban_mask.sum())]
colors_pie = ['#4CAF50', '#F44336']
ax_pie.pie(sizes, labels=['自然/半自然区', '城市硬化区'],
           colors=colors_pie, autopct='%1.2f%%', startangle=90,
           textprops={'fontsize': 11})
ax_pie.set_title('城市不透水层识别结果', fontsize=12, fontweight='bold')
ax_pie.text(0, -1.35, f"总有效像元：{valid_pixels:,}\n城市：{sizes[1]:,} 像元",
            ha='center', fontsize=10, color='gray')

plt.tight_layout()
out2 = os.path.join(VIS_DIR, 'Step1_02_Distributions.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"    [图2] 统计分布  → {out2}")

# ----------------------------------------------------------
# 图3：土壤参数关系分析（Ks-θs 散点 + Psi-Ks 散点）
# ----------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('Step 1：土壤水文参数空间关系分析', fontsize=14, fontweight='bold')

sample_n  = 60000
valid_idx = np.where(~nodata_mask.ravel())[0]
if len(valid_idx) > sample_n:
    chosen = np.random.choice(valid_idx, sample_n, replace=False)
else:
    chosen = valid_idx

ks_s    = ks.ravel()[chosen]
ths_s   = ths.ravel()[chosen]
psi_s   = psi_abs.ravel()[chosen]
urban_s = urban_mask.ravel()[chosen]

from scipy.stats import pearsonr

# Ks vs θs
ax = axes3[0]
ax.scatter(ks_s[~urban_s],  ths_s[~urban_s],  c='steelblue', alpha=0.25, s=2, label='自然土壤')
ax.scatter(ks_s[urban_s],   ths_s[urban_s],   c='red',       alpha=0.50, s=3, label='城市硬化')
ax.set_xlabel('Ks (mm/h)', fontsize=11)
ax.set_ylabel('θs (m³/m³)', fontsize=11)
ax.set_title('Ks vs θs', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
nat_valid = ~urban_s & np.isfinite(ks_s) & np.isfinite(ths_s)
if nat_valid.sum() > 3:
    r1, p1 = pearsonr(ks_s[nat_valid], ths_s[nat_valid])
    ax.text(0.05, 0.95, f"r={r1:.4f}  P={p1:.2e}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Ks vs |Psi|
ax = axes3[1]
ax.scatter(ks_s[~urban_s],  psi_s[~urban_s],  c='steelblue', alpha=0.25, s=2, label='自然土壤')
ax.scatter(ks_s[urban_s],   psi_s[urban_s],   c='red',       alpha=0.50, s=3, label='城市硬化')
ax.set_xlabel('Ks (mm/h)', fontsize=11)
ax.set_ylabel('|Psi| (cm)', fontsize=11)
ax.set_title('Ks vs |Psi|', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
nat_valid2 = ~urban_s & np.isfinite(ks_s) & np.isfinite(psi_s)
if nat_valid2.sum() > 3:
    r2, p2 = pearsonr(ks_s[nat_valid2], psi_s[nat_valid2])
    ax.text(0.05, 0.95, f"r={r2:.4f}  P={p2:.2e}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# θs vs |Psi|
ax = axes3[2]
ax.scatter(ths_s[~urban_s], psi_s[~urban_s],  c='steelblue', alpha=0.25, s=2, label='自然土壤')
ax.scatter(ths_s[urban_s],  psi_s[urban_s],   c='red',       alpha=0.50, s=3, label='城市硬化')
ax.set_xlabel('θs (m³/m³)', fontsize=11)
ax.set_ylabel('|Psi| (cm)', fontsize=11)
ax.set_title('θs vs |Psi|', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
nat_valid3 = ~urban_s & np.isfinite(ths_s) & np.isfinite(psi_s)
if nat_valid3.sum() > 3:
    r3, p3 = pearsonr(ths_s[nat_valid3], psi_s[nat_valid3])
    ax.text(0.05, 0.95, f"r={r3:.4f}  P={p3:.2e}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
out3 = os.path.join(VIS_DIR, 'Step1_03_SoilAnalysis.png')
plt.savefig(out3, dpi=200, bbox_inches='tight')
plt.close()
print(f"    [图3] 土壤分析  → {out3}")

# ----------------------------------------------------------
# 图4：汇流累积量对数图（水系提取验证）
# ----------------------------------------------------------
fig4, ax4_main = plt.subplots(figsize=(12, 8))
acc_log_disp = np.where(nodata_mask, np.nan, np.log1p(acc_arr))
im_acc = ax4_main.imshow(acc_log_disp, cmap='Blues')
river_mask = (acc_arr > 500) & ~nodata_mask
ax4_main.imshow(np.where(river_mask, 1.0, np.nan), cmap='Reds', alpha=0.8, vmin=0, vmax=1)
ax4_main.set_title('汇流累积量（对数）+ 提取水系（红色，阈值>500像元）',
                   fontsize=13, fontweight='bold')
ax4_main.axis('off')
plt.colorbar(im_acc, ax=ax4_main, shrink=0.8, label='log(汇流量+1)')
river_pct = float(river_mask.sum()) / valid_pixels * 100
ax4_main.text(0.02, 0.02,
              f"水系像元: {river_mask.sum():,} ({river_pct:.2f}%)",
              transform=ax4_main.transAxes, fontsize=10,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
out4 = os.path.join(VIS_DIR, 'Step1_04_Accumulation.png')
plt.savefig(out4, dpi=200, bbox_inches='tight')
plt.close()
print(f"    [图4] 水系提取  → {out4}")

# ============================================================
# 统计汇总输出
# ============================================================
print("\n" + "=" * 70)
print("Step 1 统计汇总（可直接用于论文 §数据来源与预处理）")
print("=" * 70)
all_stats = [dem_stats, slope_stats, hand_stats, ks_stats, ths_stats, psi_stats]
header = (f"{'指标':<22}{'有效像元':>10}{'Min':>10}{'Max':>10}"
          f"{'Mean':>10}{'Std':>10}{'CV':>8}{'偏度':>8}")
print(header)
print("-" * 95)
for s in all_stats:
    print(f"{s['name']:<22}{s['count']:>10,}{s['min']:>10.4f}{s['max']:>10.4f}"
          f"{s['mean']:>10.4f}{s['std']:>10.4f}{s['cv']:>8.4f}{s['skew']:>8.4f}")

print(f"\n✅ Step 1 完成！")
print(f"   NPY特征文件  → {OUTPUT_DIR}")
print(f"     fdir / slope / hand / acc / nodata_mask")
print(f"     ks / ths / psi / urban_mask  (共9个)")
print(f"   可视化图像   → {VIS_DIR}  （共4张）")
