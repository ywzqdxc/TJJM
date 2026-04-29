"""
Step 1: 静态下垫面预处理 (融合高精度土壤水文参数)
=====================================
提取流向(fdir)、坡度(Slope)、HAND。
处理 Ks、Theta_s、Psi，识别城市不透水层，生成物理下垫面。

修订说明 (v2.5):
  - 基于土壤数据覆盖范围创建北京轮廓掩膜（无需shp文件）
  - 所有地图北京外部完全透明
  - 城市不透水层 = 北京外部（有DEM但无土壤数据）
  - 六图总览（2×3）
"""

import numpy as np
import rasterio
from pysheds.grid import Grid
from scipy.ndimage import sobel, zoom, gaussian_filter
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['mathtext.default'] = 'regular'

# ============================================================
# 路径配置
# ============================================================
DEM_PATH   = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'

# 裁剪+重采样后的土壤参数 TIF
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

    invalid = np.zeros(raw.shape, dtype=bool)
    if nodata_val is not None:
        if nodata_val < nodata_threshold:
            invalid |= (raw < nodata_threshold)
        else:
            invalid |= (raw == nodata_val)
    else:
        invalid |= (raw < nodata_threshold)
    invalid |= ~np.isfinite(raw)

    data_clean = np.where(invalid, np.nan, raw)
    return data_clean, ~invalid


def imshow_masked(ax, data, nodata_mask, cmap, vmin=None, vmax=None, max_pixels=2000, **kwargs):
    """
    只显示有效区域，外部透明。
    nodata_mask: True=无效/外部区域
    """
    h_orig, w_orig = data.shape

    # 降采样以节省内存
    if max(h_orig, w_orig) > max_pixels:
        scale = max_pixels / max(h_orig, w_orig)
        d = zoom(data, scale, order=1).astype(np.float32)
        m = zoom((~nodata_mask).astype(np.float32), scale, order=0).astype(bool)
    else:
        d = data.copy().astype(np.float32)
        m = ~nodata_mask  # m=True表示有效区域

    # 将无效区域设为NaN
    d[~m] = np.nan

    # 计算显示范围
    valid = d[m & ~np.isnan(d)]
    if vmin is None:
        vmin = np.nanpercentile(valid, 2) if valid.size > 0 else 0
    if vmax is None:
        vmax = np.nanpercentile(valid, 98) if valid.size > 0 else 1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    # 创建全透明背景
    h_disp, w_disp = d.shape
    rgba = np.zeros((h_disp, w_disp, 4), dtype=np.float32)

    # 只给有效区域着色
    valid_pixels = m & ~np.isnan(d)
    if valid_pixels.any():
        rgba[valid_pixels] = cm(norm(d[valid_pixels]))

    im = ax.imshow(rgba, interpolation='bilinear', **kwargs)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    return sm


def print_stats(name, arr):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {'name': name, 'count': 0, 'min': np.nan, 'max': np.nan,
                'mean': np.nan, 'std': np.nan, 'cv': np.nan}
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
nodata_mask_dem = np.isnan(dem_clean)
valid_pixels = int(np.sum(~nodata_mask_dem))

print(f"    栅格尺寸: {h} 行 × {w} 列")
print(f"    DEM有效像元: {valid_pixels:,}  /  总像元: {h * w:,}")

dem_stats = print_stats('DEM高程(m)', dem_clean)

# ============================================================
# 2/4  计算基础地形水文特征
# ============================================================
print("\n[2/4] 计算基础地形水文特征...")

# 坡度
dy = sobel(dem_clean, axis=0) / 30.0
dx = sobel(dem_clean, axis=1) / 30.0
slope_arr = np.clip(np.rad2deg(np.arctan(np.sqrt(dx**2 + dy**2))), 0.01, 60.0)
slope_arr = np.where(nodata_mask_dem, np.nan, slope_arr)

# 流向
pit_filled = grid.fill_pits(dem)
flooded    = grid.fill_depressions(pit_filled)
inflated   = grid.resolve_flats(flooded)
dirmap     = (64, 128, 1, 2, 4, 8, 16, 32)
fdir       = grid.flowdir(inflated, dirmap=dirmap)

# 汇流累积量
acc          = grid.accumulation(fdir, dirmap=dirmap)
channel_mask = acc > 500

# HAND
try:
    hand     = grid.compute_hand(fdir, inflated, channel_mask, dirmap=dirmap)
    hand_arr = np.clip(np.array(hand, dtype=np.float32), 0, 500)
    print("    HAND 计算成功（pysheds 原生）")
except Exception as e:
    print(f"    HAND 计算回退至归一化DEM代理 ({e})")
    dem_norm = (dem_clean - np.nanmin(dem_clean)) / (np.nanmax(dem_clean) - np.nanmin(dem_clean) + 1e-8)
    hand_arr = dem_norm * 100.0

hand_arr  = np.where(nodata_mask_dem, np.nan, hand_arr)
fdir_arr  = np.array(fdir, dtype=np.float32)
acc_arr   = np.array(acc,  dtype=np.float32)

slope_stats = print_stats('坡度(°)', slope_arr)
hand_stats  = print_stats('HAND(m)', hand_arr)

# ============================================================
# 3/4  读取土壤物理参数并识别城市不透水层
# ============================================================
print("\n[3/4] 解析土壤物理参数 (Ks, θs, Psi) 并识别城市硬化面...")

ks_raw,  ks_valid_mask  = read_tif_aligned(KS_PATH,  h, w)
ths_raw, ths_valid_mask = read_tif_aligned(THS_PATH, h, w)
psi_raw, psi_valid_mask = read_tif_aligned(PSI_PATH, h, w)

print(f"    原始 Ks  有效值范围: [{np.nanmin(ks_raw):.4f}, {np.nanmax(ks_raw):.4f}] mm/h")
print(f"    原始 θs  有效值范围: [{np.nanmin(ths_raw):.4f}, {np.nanmax(ths_raw):.4f}] m^3/m^3")
print(f"    原始 Psi 有效值范围: [{np.nanmin(psi_raw):.4f}, {np.nanmax(psi_raw):.4f}] cm")

# ---- 创建北京轮廓掩膜 ----
print("\n  >>> 创建北京轮廓掩膜...")
# 土壤数据覆盖的区域 = 自然地表（北京市内）
beijing_inside_mask = ks_valid_mask & ths_valid_mask & ~nodata_mask_dem
# 北京外部 = DEM有效但土壤无效 = 城市建成区
beijing_outside_mask = ~beijing_inside_mask & ~nodata_mask_dem
# 完整nodata掩膜（城市区+DEM nodata）
nodata_mask_full = nodata_mask_dem | beijing_outside_mask

beijing_inside = int(beijing_inside_mask.sum())
beijing_outside = int(beijing_outside_mask.sum())
print(f"    自然地表像元: {beijing_inside:,}")
print(f"    城市建成区像元: {beijing_outside:,}")
print(f"    总透明像元: {int(nodata_mask_full.sum()):,}")

# ---- 城市不透水层识别 ----
# 城市 = 北京外部（有DEM但无土壤数据）
urban_mask = beijing_outside_mask
urban_count = int(urban_mask.sum())
urban_pct = float(urban_count) / valid_pixels * 100
print(f"\n    城市不透水层: {urban_count:,} 像元 ({urban_pct:.1f}% 全域面积)")

# ---- 物理参数赋值 ----
KS_URBAN   = 0.01    # mm/h  (几乎不透水)
THS_URBAN  = 0.10    # m^3/m^3 (极低含水量)
PSI_URBAN  = -3.00   # cm    (极小毛管吸力)

# Ks
ks = ks_raw.copy()
ks[urban_mask]  = KS_URBAN
ks = np.clip(ks, 0.01, 200.0)
ks = np.where(nodata_mask_dem, np.nan, ks)  # 只保留DEM有效区域

# θs
ths = ths_raw.copy()
ths[urban_mask] = THS_URBAN
ths = np.clip(ths, 0.05, 0.65)
ths = np.where(nodata_mask_dem, np.nan, ths)

# Psi
psi_abs = np.abs(psi_raw)
psi_abs[urban_mask] = abs(PSI_URBAN)
psi_abs = np.clip(psi_abs, 0.1, 100.0)
psi_abs = np.where(nodata_mask_dem, np.nan, psi_abs)

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
np.save(os.path.join(OUTPUT_DIR, 'nodata_mask.npy'), nodata_mask_full)
np.save(os.path.join(OUTPUT_DIR, 'ks.npy'),          ks)
np.save(os.path.join(OUTPUT_DIR, 'ths.npy'),         ths)
np.save(os.path.join(OUTPUT_DIR, 'psi.npy'),         psi_abs)
np.save(os.path.join(OUTPUT_DIR, 'urban_mask.npy'),  urban_mask)
print("\n    所有特征已保存至 NPY（共9个）。")

# ============================================================
# 4/4  完整可视化
# ============================================================
print("\n[4/4] 生成可视化图像（北京轮廓透明背景）...")

# ----------------------------------------------------------
# 图1：六图总览（2×3）
# ----------------------------------------------------------
fig = plt.figure(figsize=(18, 12))
fig.patch.set_alpha(0)
fig.suptitle('Step 1：北京市静态下垫面特征提取总览\n(30m NASA DEM + 土壤水文参数 Ks/θs/Psi)',
             fontsize=15, fontweight='bold', y=0.99)

gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
              width_ratios=[1, 1, 1], height_ratios=[1, 1])

panels = [
    (fig.add_subplot(gs[0, 0]), dem_clean,  'terrain',    '① DEM 高程 (m)',            dem_stats),
    (fig.add_subplot(gs[0, 1]), slope_arr,  'RdYlGn_r',   '② 坡度 (°)',                slope_stats),
    (fig.add_subplot(gs[0, 2]), hand_arr,   'Blues_r',    '③ HAND (m)',                hand_stats),
    (fig.add_subplot(gs[1, 0]), ks,         'YlOrRd',     '④ 饱和导水率 Ks (mm/h)',    ks_stats),
    (fig.add_subplot(gs[1, 1]), ths,        'BrBG',       '⑤ 饱和含水量 θs',          ths_stats),
    (fig.add_subplot(gs[1, 2]), psi_abs,    'PuBu',       '⑥ 毛管吸力 |Psi| (cm)',    psi_stats),
]

for ax, data, cmap, title, st in panels:
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
    sm = imshow_masked(ax, data, nodata_mask_full, cmap)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    cb = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, aspect=20)
    cb.set_label(title.split(' ')[0], fontsize=8)
    ax.text(0.02, 0.02,
            f"均={st['mean']:.3f}\nCV={st['cv']:.3f}",
            transform=ax.transAxes, fontsize=8.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                    wspace=0.30, hspace=0.35)

out1 = os.path.join(VIS_DIR, 'Step1_01_Overview.png')
plt.savefig(out1, dpi=200, bbox_inches='tight', transparent=True)
plt.close()
print(f"    [图1] 六图总览  -> {out1}")

# ----------------------------------------------------------
# 图2：统计分布直方图
# ----------------------------------------------------------
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.patch.set_alpha(0)
fig2.suptitle('Step 1：关键地形与土壤特征统计分布', fontsize=14, fontweight='bold')

hist_configs = [
    (slope_arr, '坡度 (°)',       'steelblue',   slope_stats),
    (hand_arr,  'HAND (m)',       'seagreen',    hand_stats),
    (ks,        'Ks (mm/h)',      'darkorange',  ks_stats),
    (ths,       'θs (m^3/m^3)',  'saddlebrown', ths_stats),
    (psi_abs,   '|Psi| (cm)',     'mediumpurple',psi_stats),
]
for ax, (data, label, color, st) in zip(axes2.flat, hist_configs):
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0:
        continue
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
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# 第6图：自然区/城市区饼图
ax_pie = axes2.flat[5]
sizes  = [beijing_inside, beijing_outside]
labels_pie = ['自然地表', '城市建成区']
colors_pie = ['#4CAF50', '#E91E63']
ax_pie.pie(sizes, labels=labels_pie,
           colors=colors_pie, autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11})
ax_pie.set_title('城市不透水层识别结果\n(无土壤数据→城市)', fontsize=12, fontweight='bold')
ax_pie.text(0, -1.35, f"全域有效像元：{valid_pixels:,}\n自然：{beijing_inside:,}  城市：{beijing_outside:,}",
            ha='center', fontsize=10, color='gray')

plt.tight_layout()
out2 = os.path.join(VIS_DIR, 'Step1_02_Distributions.png')
plt.savefig(out2, dpi=200, bbox_inches='tight', transparent=True)
plt.close()
print(f"    [图2] 统计分布  -> {out2}")

# ----------------------------------------------------------
# 图3：土壤参数关系分析（hexbin热力图）
# ----------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.patch.set_alpha(0)
fig3.suptitle('Step 1：土壤水文参数空间关系分析', fontsize=14, fontweight='bold')

sample_n  = 100000
valid_idx = np.where(~nodata_mask_full.ravel())[0]
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
nat_idx = ~urban_s & np.isfinite(ks_s) & np.isfinite(ths_s)
if nat_idx.sum() > 10:
    hb1 = ax.hexbin(ks_s[nat_idx], ths_s[nat_idx], gridsize=40, cmap='Blues',
                     mincnt=1, bins='log')
    plt.colorbar(hb1, ax=ax, shrink=0.8, label='自然土壤: log10(计数)')
    r1, p1 = pearsonr(ks_s[nat_idx][:min(50000, nat_idx.sum())],
                       ths_s[nat_idx][:min(50000, nat_idx.sum())])
    ax.text(0.05, 0.95, f"r={r1:.4f}  P={p1:.2e}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
urb_idx = urban_s & np.isfinite(ks_s) & np.isfinite(ths_s)
if urb_idx.sum() > 0:
    ax.scatter(ks_s[urb_idx], ths_s[urb_idx], c='#E91E63', alpha=0.5, s=5, label='城市')
ax.set_xlabel('Ks (mm/h)', fontsize=11)
ax.set_ylabel('θs (m^3/m^3)', fontsize=11)
ax.set_title('Ks vs θs', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# Ks vs |Psi|
ax = axes3[1]
nat_idx2 = ~urban_s & np.isfinite(ks_s) & np.isfinite(psi_s)
if nat_idx2.sum() > 10:
    hb2 = ax.hexbin(ks_s[nat_idx2], psi_s[nat_idx2], gridsize=40, cmap='Blues',
                     mincnt=1, bins='log')
    plt.colorbar(hb2, ax=ax, shrink=0.8, label='自然土壤: log10(计数)')
    r2, p2 = pearsonr(ks_s[nat_idx2][:min(50000, nat_idx2.sum())],
                       psi_s[nat_idx2][:min(50000, nat_idx2.sum())])
    ax.text(0.05, 0.95, f"r={r2:.4f}  P={p2:.2e}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
urb_idx2 = urban_s & np.isfinite(ks_s) & np.isfinite(psi_s)
if urb_idx2.sum() > 0:
    ax.scatter(ks_s[urb_idx2], psi_s[urb_idx2], c='#E91E63', alpha=0.5, s=5, label='城市')
ax.set_xlabel('Ks (mm/h)', fontsize=11)
ax.set_ylabel('|Psi| (cm)', fontsize=11)
ax.set_title('Ks vs |Psi|', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# θs vs |Psi|
ax = axes3[2]
nat_idx3 = ~urban_s & np.isfinite(ths_s) & np.isfinite(psi_s)
if nat_idx3.sum() > 10:
    hb3 = ax.hexbin(ths_s[nat_idx3], psi_s[nat_idx3], gridsize=40, cmap='Blues',
                     mincnt=1, bins='log')
    plt.colorbar(hb3, ax=ax, shrink=0.8, label='自然土壤: log10(计数)')
    r3, p3 = pearsonr(ths_s[nat_idx3][:min(50000, nat_idx3.sum())],
                       psi_s[nat_idx3][:min(50000, nat_idx3.sum())])
    ax.text(0.05, 0.95, f"r={r3:.4f}  P={p3:.2e}", transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
urb_idx3 = urban_s & np.isfinite(ths_s) & np.isfinite(psi_s)
if urb_idx3.sum() > 0:
    ax.scatter(ths_s[urb_idx3], psi_s[urb_idx3], c='#E91E63', alpha=0.5, s=5, label='城市')
ax.set_xlabel('θs (m^3/m^3)', fontsize=11)
ax.set_ylabel('|Psi| (cm)', fontsize=11)
ax.set_title('θs vs |Psi|', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
out3 = os.path.join(VIS_DIR, 'Step1_03_SoilAnalysis.png')
plt.savefig(out3, dpi=200, bbox_inches='tight', transparent=True)
plt.close()
print(f"    [图3] 土壤分析  -> {out3}")

# ----------------------------------------------------------
# 图4：汇流累积量对数图
# ----------------------------------------------------------
fig4, ax4_main = plt.subplots(figsize=(12, 8))
fig4.patch.set_alpha(0)
ax4_main.set_facecolor('none')
ax4_main.patch.set_alpha(0)
sm_acc = imshow_masked(ax4_main, np.log1p(acc_arr), nodata_mask_full, 'Blues')
river_mask = (acc_arr > 500) & ~nodata_mask_full
ax4_main.imshow(np.where(river_mask, 1.0, np.nan), cmap='Reds', alpha=0.8, vmin=0, vmax=1)
ax4_main.set_title('汇流累积量（对数）+ 提取水系（红色，阈值>500像元）',
                   fontsize=13, fontweight='bold')
ax4_main.axis('off')
plt.colorbar(sm_acc, ax=ax4_main, shrink=0.8, pad=0.02, label='log(acc+1)')
river_pct = float(river_mask.sum()) / valid_pixels * 100
ax4_main.text(0.02, 0.02,
              f"水系像元: {river_mask.sum():,} ({river_pct:.2f}%)",
              transform=ax4_main.transAxes, fontsize=10,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))
plt.tight_layout()
out4 = os.path.join(VIS_DIR, 'Step1_04_Accumulation.png')
plt.savefig(out4, dpi=200, bbox_inches='tight', transparent=True)
plt.close()
print(f"    [图4] 水系提取  -> {out4}")

# ============================================================
# 统计汇总输出
# ============================================================
print("\n" + "=" * 70)
print("Step 1 统计汇总")
print("=" * 70)
print(f"  全域有效像元: {valid_pixels:,}")
print(f"  自然地表:     {beijing_inside:,} 像元 ({beijing_inside/valid_pixels*100:.1f}%)")
print(f"  城市建成区:   {beijing_outside:,} 像元 ({urban_pct:.1f}%)")
print(f"  水系像元:     {int(river_mask.sum()):,} 像元 ({river_pct:.2f}%)")
all_stats = [dem_stats, slope_stats, hand_stats, ks_stats, ths_stats, psi_stats]
header = (f"{'指标':<22}{'有效像元':>10}{'Min':>10}{'Max':>10}"
          f"{'Mean':>10}{'Std':>10}{'CV':>8}{'偏度':>8}")
print(header)
print("-" * 95)
for s in all_stats:
    print(f"{s['name']:<22}{s['count']:>10,}{s['min']:>10.4f}{s['max']:>10.4f}"
          f"{s['mean']:>10.4f}{s['std']:>10.4f}{s['cv']:>8.4f}{s['skew']:>8.4f}")

print(f"\n✅ Step 1 完成！")
print(f"   NPY特征文件  -> {OUTPUT_DIR}")
print(f"     fdir / slope / hand / acc / nodata_mask")
print(f"     ks / ths / psi / urban_mask  (共9个)")
print(f"   可视化图像   -> {VIS_DIR}  （共4张）")