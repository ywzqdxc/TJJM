"""
Step 1: 静态下垫面预处理 (融合高精度土壤水文参数)
=====================================
提取流向(fdir)、坡度(Slope)、HAND。
处理 Ks 和 Theta_s，识别城市不透水层，生成物理下垫面。
新增：完整过程可视化 + 统计描述输出
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
KS_PATH    = r'F:\Data\src\fac\Aligned_Soil_Params\K_SCH_Aligned_30m.tif'
THS_PATH   = r'F:\Data\src\fac\Aligned_Soil_Params\THSCH_Aligned_30m.tif'
OUTPUT_DIR = r'./Step_New/Static'
VIS_DIR    = r'./Step_New/Visualization/Step1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,    exist_ok=True)

# ============================================================
# 工具函数
# ============================================================
def safe_percentile(arr, pct, ignore_nan=True):
    """安全的百分位数计算，忽略 nan"""
    if ignore_nan:
        return float(np.nanpercentile(arr, pct))
    return float(np.percentile(arr, pct))

def print_stats(name, arr):
    """打印并返回一组描述性统计"""
    valid = arr[~np.isnan(arr)]
    stats_dict = {
        'name': name,
        'count': int(valid.size),
        'min':   float(np.min(valid)),
        'max':   float(np.max(valid)),
        'mean':  float(np.mean(valid)),
        'std':   float(np.std(valid)),
        'cv':    float(np.std(valid) / (np.mean(valid) + 1e-10)),
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
    """截断极值后用于显示"""
    lo_v = safe_percentile(arr[~np.isnan(arr)], lo)
    hi_v = safe_percentile(arr[~np.isnan(arr)], hi)
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
    profile = ref.profile.copy()
    h, w    = ref.height, ref.width
    transform = ref.transform
    crs       = ref.crs

dem_clean    = np.where(dem < -100, np.nan, dem.astype(np.float32))
nodata_mask  = np.isnan(dem_clean)
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
acc = grid.accumulation(fdir, dirmap=dirmap)
channel_mask = acc > 500

# --- HAND ---
try:
    hand = grid.compute_hand(fdir, inflated, channel_mask, dirmap=dirmap)
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

slope_stats = print_stats('坡度(°)',    slope_arr)
hand_stats  = print_stats('HAND(m)',    hand_arr)

# ============================================================
# 3/4  解析土壤物理参数
# ============================================================
print("\n[3/4] 解析土壤物理参数 (Ks & θs) 并识别城市硬化面...")

# --- θs 饱和含水量 ---
with rasterio.open(THS_PATH) as src:
    ths_raw = src.read(1, out_shape=(h, w)).astype(np.float32)

urban_mask = (ths_raw < 0) | (ths_raw > 1) | nodata_mask
ths        = ths_raw.copy()
ths[urban_mask & ~nodata_mask] = 0.10
ths        = np.clip(ths, 0.10, 0.60)
ths        = np.where(nodata_mask, np.nan, ths)

# --- Ks 饱和导水率 ---
with rasterio.open(KS_PATH) as src:
    ks_raw = src.read(1, out_shape=(h, w)).astype(np.float32)

ks = ks_raw.copy()
ks[urban_mask & ~nodata_mask] = 0.01
ks = np.clip(ks, 0.01, 200.0)
ks = np.where(nodata_mask, np.nan, ks)

urban_pct = float(np.sum(urban_mask & ~nodata_mask)) / valid_pixels * 100
print(f"    识别城市不透水层像元: {int(np.sum(urban_mask & ~nodata_mask)):,}  ({urban_pct:.2f}% 全域面积)")

ks_stats  = print_stats('饱和导水率Ks(mm/h)', ks)
ths_stats = print_stats('饱和含水量θs',       ths)

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
np.save(os.path.join(OUTPUT_DIR, 'urban_mask.npy'),  urban_mask)
print("\n    所有特征已保存至 NPY。")

# ============================================================
# 4/4  完整可视化
# ============================================================
print("\n[4/4] 生成可视化图像...")

# ----------------------------------------------------------
# 图1：六图总览
# ----------------------------------------------------------
fig = plt.figure(figsize=(22, 14))
fig.suptitle('Step 1：北京市静态下垫面特征提取总览\n(基于30m NASA DEM + 土壤水文参数)',
             fontsize=16, fontweight='bold', y=0.98)

gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# -- DEM --
ax0 = fig.add_subplot(gs[0, 0])
dem_disp = clip_for_display(dem_clean)
im0 = ax0.imshow(dem_disp, cmap='terrain')
ax0.set_title('① DEM 高程 (m)', fontsize=12, fontweight='bold')
ax0.axis('off')
cb0 = plt.colorbar(im0, ax=ax0, shrink=0.85)
cb0.set_label('高程 (m)')
ax0.text(0.02, 0.02, f"均值={dem_stats['mean']:.1f}m\nStd={dem_stats['std']:.1f}m",
         transform=ax0.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# -- 流向 fdir --
ax1 = fig.add_subplot(gs[0, 1])
fdir_disp = np.where(nodata_mask, np.nan, fdir_arr)
im1 = ax1.imshow(fdir_disp, cmap='tab20b', interpolation='nearest')
ax1.set_title('② D8 流向 (fdir)', fontsize=12, fontweight='bold')
ax1.axis('off')
cb1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
cb1.set_label('流向编码')
# 流向图例说明
dir_labels = {64:'NW', 128:'N', 1:'NE', 2:'E', 4:'SE', 8:'S', 16:'SW', 32:'W'}
legend_txt = '  '.join([f"{k}={v}" for k, v in dir_labels.items()])
ax1.text(0.5, -0.04, legend_txt, transform=ax1.transAxes, fontsize=7.5,
         ha='center', color='gray')

# -- 坡度 --
ax2 = fig.add_subplot(gs[0, 2])
slope_disp = clip_for_display(slope_arr)
im2 = ax2.imshow(slope_disp, cmap='RdYlGn_r')
ax2.set_title('③ 坡度 (°)', fontsize=12, fontweight='bold')
ax2.axis('off')
cb2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
cb2.set_label('坡度 (°)')
ax2.text(0.02, 0.02, f"均值={slope_stats['mean']:.2f}°\nCV={slope_stats['cv']:.3f}",
         transform=ax2.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# -- HAND --
ax3 = fig.add_subplot(gs[1, 0])
hand_disp = clip_for_display(hand_arr, hi=95)
im3 = ax3.imshow(hand_disp, cmap='Blues_r')
ax3.set_title('④ HAND（高于最近水道高程，m）', fontsize=12, fontweight='bold')
ax3.axis('off')
cb3 = plt.colorbar(im3, ax=ax3, shrink=0.85)
cb3.set_label('HAND (m)')
ax3.text(0.02, 0.02, f"均值={hand_stats['mean']:.1f}m\nP50={hand_stats['p50']:.1f}m",
         transform=ax3.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# -- Ks --
ax4 = fig.add_subplot(gs[1, 1])
ks_disp = clip_for_display(ks)
im4 = ax4.imshow(ks_disp, cmap='YlOrRd')
ax4.set_title('⑤ 饱和导水率 Ks (mm/h)', fontsize=12, fontweight='bold')
ax4.axis('off')
cb4 = plt.colorbar(im4, ax=ax4, shrink=0.85)
cb4.set_label('Ks (mm/h)')
ax4.text(0.02, 0.02, f"均值={ks_stats['mean']:.1f}\nCV={ks_stats['cv']:.3f}",
         transform=ax4.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# -- θs + 城市掩膜 --
ax5 = fig.add_subplot(gs[1, 2])
ths_disp = clip_for_display(ths)
im5 = ax5.imshow(ths_disp, cmap='BrBG')
# 城市区叠加高亮
urban_overlay = np.where(urban_mask & ~nodata_mask, 1.0, np.nan)
ax5.imshow(urban_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
ax5.set_title('⑥ 饱和含水量 θs（红色=城市硬化）', fontsize=12, fontweight='bold')
ax5.axis('off')
cb5 = plt.colorbar(im5, ax=ax5, shrink=0.85)
cb5.set_label('θs (m³/m³)')
ax5.text(0.02, 0.02, f"城市占比={urban_pct:.1f}%\nθs均值={ths_stats['mean']:.3f}",
         transform=ax5.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

out1 = os.path.join(VIS_DIR, 'Step1_01_Overview.png')
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"    [图1] 六图总览  → {out1}")

# ----------------------------------------------------------
# 图2：统计分布直方图（4个关键指标）
# ----------------------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Step 1：关键地形与土壤特征统计分布', fontsize=14, fontweight='bold')

hist_configs = [
    (slope_arr, '坡度 (°)',      'steelblue',  slope_stats),
    (hand_arr,  'HAND (m)',      'seagreen',   hand_stats),
    (ks,        'Ks (mm/h)',     'darkorange', ks_stats),
    (ths,       'θs (m³/m³)',    'saddlebrown',ths_stats),
]
for ax, (data, label, color, st) in zip(axes2.flat, hist_configs):
    valid_data = data[~np.isnan(data)]
    # 截断99.5%分位以避免极值干扰
    upper = np.percentile(valid_data, 99.5)
    d_trim = valid_data[valid_data <= upper]
    ax.hist(d_trim, bins=80, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(st['mean'], color='red',    linestyle='--', linewidth=1.5, label=f"均值={st['mean']:.3f}")
    ax.axvline(st['p50'],  color='orange', linestyle='-',  linewidth=1.5, label=f"中位数={st['p50']:.3f}")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('像元频数', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    stats_text = (f"N={st['count']:,}\nStd={st['std']:.4f}\n"
                  f"CV={st['cv']:.4f}\n偏度={st['skew']:.3f}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
out2 = os.path.join(VIS_DIR, 'Step1_02_Distributions.png')
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"    [图2] 统计分布  → {out2}")

# ----------------------------------------------------------
# 图3：Ks 与 θs 散点图（城市/非城市分色）
# ============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('Step 1：土壤参数空间关系分析', fontsize=14, fontweight='bold')

# 采样以避免绘制全部像元
sample_n = 50000
valid_idx = np.where(~nodata_mask.ravel())[0]
if len(valid_idx) > sample_n:
    chosen = np.random.choice(valid_idx, sample_n, replace=False)
else:
    chosen = valid_idx

ks_s    = ks.ravel()[chosen]
ths_s   = ths.ravel()[chosen]
urban_s = urban_mask.ravel()[chosen]

ax = axes3[0]
ax.scatter(ks_s[~urban_s],  ths_s[~urban_s],  c='steelblue',  alpha=0.3, s=2, label='自然土壤')
ax.scatter(ks_s[urban_s],   ths_s[urban_s],   c='red',        alpha=0.6, s=4, label='城市硬化')
ax.set_xlabel('饱和导水率 Ks (mm/h)', fontsize=11)
ax.set_ylabel('饱和含水量 θs (m³/m³)', fontsize=11)
ax.set_title('Ks vs θs（50,000点采样）', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# Pearson相关（仅自然土壤）
from scipy.stats import pearsonr
nat_ks  = ks_s[~urban_s]
nat_ths = ths_s[~urban_s]
r_val, p_val = pearsonr(nat_ks, nat_ths)
ax.text(0.05, 0.95, f"Pearson r={r_val:.4f}\nP值={p_val:.4e}",
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 城市区面积饼图
ax2_r = axes3[1]
labels = ['自然/半自然区', '城市硬化区']
sizes  = [(~urban_mask & ~nodata_mask).sum(), (urban_mask & ~nodata_mask).sum()]
colors = ['#4CAF50', '#F44336']
wedges, texts, autotexts = ax2_r.pie(
    sizes, labels=labels, colors=colors, autopct='%1.2f%%',
    startangle=90, textprops={'fontsize': 11}
)
ax2_r.set_title('城市不透水层识别结果', fontsize=12, fontweight='bold')
ax2_r.text(0, -1.3, f"总有效像元：{valid_pixels:,}\n城市硬化：{sizes[1]:,} 像元",
           ha='center', fontsize=10, color='gray')

plt.tight_layout()
out3 = os.path.join(VIS_DIR, 'Step1_03_SoilAnalysis.png')
plt.savefig(out3, dpi=200, bbox_inches='tight')
plt.close()
print(f"    [图3] 土壤分析  → {out3}")

# ----------------------------------------------------------
# 图4：汇流累积量对数图（水系提取验证）
# ----------------------------------------------------------
fig4, ax4_main = plt.subplots(figsize=(12, 8))
acc_log = np.log1p(acc_arr)
acc_log_disp = np.where(nodata_mask, np.nan, acc_log)
im_acc = ax4_main.imshow(acc_log_disp, cmap='Blues')
# 叠加主河道（acc > 500）
river_mask = (acc_arr > 500) & ~nodata_mask
river_overlay = np.where(river_mask, 1.0, np.nan)
ax4_main.imshow(river_overlay, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
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
# 统计汇总输出（用于论文数据说明表）
# ============================================================
print("\n" + "=" * 70)
print("Step 1 统计汇总（可直接用于论文 §数据来源与预处理）")
print("=" * 70)
all_stats = [dem_stats, slope_stats, hand_stats, ks_stats, ths_stats]
header = f"{'指标':<20}{'有效像元':>10}{'Min':>10}{'Max':>10}{'Mean':>10}{'Std':>10}{'CV':>8}{'偏度':>8}"
print(header)
print("-" * 90)
for s in all_stats:
    print(f"{s['name']:<20}{s['count']:>10,}{s['min']:>10.4f}{s['max']:>10.4f}"
          f"{s['mean']:>10.4f}{s['std']:>10.4f}{s['cv']:>8.4f}{s['skew']:>8.4f}")

print(f"\n✅ Step 1 完成！")
print(f"   NPY特征文件  → {OUTPUT_DIR}")
print(f"   可视化图像   → {VIS_DIR}  （共4张）")