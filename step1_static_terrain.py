"""
Step 1 (v5.0): 静态下垫面预处理
=================================
v5.0 相对 v3.0 的唯一改动：
  在"保存NPY"区域新增 acc.npy 显式保存
  （汇流累积量，Step3 TWI计算的必要输入）

其余逻辑与 v3.0 完全一致：
  - 三类下垫面划分（城市/裸岩/自然）
  - SCS-CN产流系数参数赋值
  - pysheds 流向+汇流累积量计算
"""

import numpy as np
import rasterio
from pysheds.grid import Grid
from scipy.ndimage import sobel, zoom, gaussian_filter
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
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
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
KS_PATH    = r'E:\Data\src\fac\KSCH\ksch\Extract_K_SC1_Resample1.tif'
THS_PATH   = r'E:\Data\src\fac\THSCH\thsch\Extract_THSC1_Resample1.tif'
PSI_PATH   = r'E:\Data\src\fac\PSI\psi\Extract_PSI_1_Resample1.tif'

OUTPUT_DIR = r'./Step_New/Static'
VIS_DIR    = r'./Step_New/Visualization/Step1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,    exist_ok=True)

# ============================================================
# 三类下垫面参数（SCS-CN经验值）
# ============================================================
KS_URBAN   = 0.01    # mm/h   城市建成区
THS_URBAN  = 0.10
PSI_URBAN  = 3.00

KS_ROCK    = 2.00    # mm/h   山区裸岩区
THS_ROCK   = 0.15
PSI_ROCK   = 5.00

ROCK_SLOPE_THRESHOLD = 25.0   # 度，坡度>25°且土壤参数缺失→裸岩

NODATA_THRESHOLD = -1e30
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

# ============================================================
# 工具函数
# ============================================================
def read_tif_aligned(path, h, w, nodata_threshold=NODATA_THRESHOLD):
    with rasterio.open(path) as src:
        raw = src.read(
            1, out_shape=(h, w),
            resampling=rasterio.enums.Resampling.bilinear
        ).astype(np.float32)
        nodata_val = src.nodata
    invalid = np.zeros(raw.shape, dtype=bool)
    if nodata_val is not None:
        invalid |= (raw == nodata_val) if abs(nodata_val) < 1e10 \
                   else (raw < nodata_threshold)
    invalid |= ~np.isfinite(raw)
    return np.where(invalid, np.nan, raw), ~invalid


def imshow_masked(ax, data, nodata_mask, cmap, vmin=None, vmax=None,
                  max_pixels=2000, **kwargs):
    h_o, w_o = data.shape
    if max(h_o, w_o) > max_pixels:
        sc = max_pixels / max(h_o, w_o)
        d  = zoom(data, sc, order=1).astype(np.float32)
        m  = zoom((~nodata_mask).astype(np.float32), sc, order=0).astype(bool)
    else:
        d, m = data.copy().astype(np.float32), ~nodata_mask
    d[~m] = np.nan
    valid = d[m & ~np.isnan(d)]
    if vmin is None: vmin = np.nanpercentile(valid, 2)  if valid.size > 0 else 0
    if vmax is None: vmax = np.nanpercentile(valid, 98) if valid.size > 0 else 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm   = plt.get_cmap(cmap)
    rgba = np.zeros((*d.shape, 4), dtype=np.float32)
    vp   = m & ~np.isnan(d)
    if vp.any(): rgba[vp] = cm(norm(d[vp]))
    ax.imshow(rgba, interpolation='bilinear', **kwargs)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
    return sm


def print_stats(name, arr):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return dict(name=name, count=0, min=np.nan, max=np.nan,
                    mean=np.nan, std=np.nan, cv=np.nan,
                    p25=np.nan, p50=np.nan, p75=np.nan,
                    skew=np.nan, kurt=np.nan)
    st = dict(
        name=name, count=int(valid.size),
        min=float(np.min(valid)),   max=float(np.max(valid)),
        mean=float(np.mean(valid)), std=float(np.std(valid)),
        cv=float(np.std(valid)/(abs(np.mean(valid))+1e-10)),
        p25=float(np.percentile(valid,25)),
        p50=float(np.percentile(valid,50)),
        p75=float(np.percentile(valid,75)),
        skew=float(stats.skew(valid)),
        kurt=float(stats.kurtosis(valid)),
    )
    print(f"\n  [{name}]")
    print(f"    有效像元: {st['count']:,}")
    print(f"    Min={st['min']:.4f}  Max={st['max']:.4f}")
    print(f"    Mean={st['mean']:.4f}  Std={st['std']:.4f}  CV={st['cv']:.4f}")
    print(f"    P25={st['p25']:.4f}  P50={st['p50']:.4f}  P75={st['p75']:.4f}")
    print(f"    偏度={st['skew']:.4f}  峰度={st['kurt']:.4f}")
    return st


# ============================================================
# 1/4  初始化 DEM 与 pysheds
# ============================================================
print("=" * 70)
print("Step 1 (v5.0)：静态下垫面预处理")
print("=" * 70)
print("\n[1/4] 初始化 DEM 与 Pysheds 物理网格...")

grid = Grid.from_raster(DEM_PATH)
dem  = grid.read_raster(DEM_PATH)

with rasterio.open(DEM_PATH) as ref:
    profile   = ref.profile.copy()
    h, w      = ref.height, ref.width
    transform = ref.transform
    crs       = ref.crs

dem_clean       = np.where(dem < -100, np.nan, dem.astype(np.float32))
nodata_mask_dem = np.isnan(dem_clean)
valid_pixels    = int(np.sum(~nodata_mask_dem))

print(f"    栅格尺寸: {h} 行 × {w} 列")
print(f"    DEM有效像元: {valid_pixels:,}  /  总像元: {h*w:,}")

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
fdir       = grid.flowdir(inflated, dirmap=DIRMAP)

# 汇流累积量
acc        = grid.accumulation(fdir, dirmap=DIRMAP)
acc_arr    = np.array(acc, dtype=np.float32)   # ★ 显式转为numpy数组备用

# HAND (Height Above Nearest Drainage)
channel_mask = acc > 500
try:
    hand_obj = grid.compute_hand(fdir, dem, channel_mask, dirmap=DIRMAP)
    hand_arr = np.array(hand_obj, dtype=np.float32)
    hand_arr = np.where(nodata_mask_dem | (hand_arr < 0), np.nan, hand_arr)
except Exception:
    # 备用方法：高斯模糊简化版
    print("    ⚠️  HAND计算降级（用平滑DEM近似）")
    dem_smooth = gaussian_filter(np.where(np.isnan(dem_clean), 0, dem_clean), sigma=5)
    hand_arr   = np.clip(dem_clean - dem_smooth, 0, None)
    hand_arr   = np.where(nodata_mask_dem, np.nan, hand_arr)

slope_stats = print_stats('坡度(°)', slope_arr)
hand_stats  = print_stats('HAND(m)',  hand_arr)

print(f"\n    汇流累积量  max={acc_arr[~nodata_mask_dem].max():.0f}"
      f"  P99={np.percentile(acc_arr[~nodata_mask_dem], 99):.1f}")


# ============================================================
# 3/4  土壤参数读取 + 三类下垫面划分
# ============================================================
print("\n[3/4] 土壤参数读取与三类下垫面划分...")

ks,  valid_ks  = read_tif_aligned(KS_PATH,  h, w)
ths, valid_ths = read_tif_aligned(THS_PATH, h, w)
psi, valid_psi = read_tif_aligned(PSI_PATH, h, w)
psi_abs = np.where(np.isnan(psi), np.nan, np.abs(psi))

# 联合有效掩膜
nodata_mask_soil = nodata_mask_dem | (np.isnan(ks) & np.isnan(ths))
nodata_mask_full = nodata_mask_dem.copy()

# ── 三类下垫面判定 ──────────────────────────────
soil_missing = np.isnan(ks) & ~nodata_mask_dem
steep_slope  = (slope_arr > ROCK_SLOPE_THRESHOLD) & ~nodata_mask_dem

# 城市建成区：土壤数据缺失 & 坡度≤25°
urban_mask   = soil_missing & ~steep_slope
# 山区裸岩区：土壤数据缺失 & 坡度>25°
rock_mask    = soil_missing & steep_slope
# 自然植被区：有实测土壤数据
natural_mask = ~soil_missing & ~nodata_mask_dem

# 各类像元统计
valid_total   = int((~nodata_mask_dem).sum())
urban_count   = int(urban_mask.sum())
rock_count    = int(rock_mask.sum())
natural_count = int(natural_mask.sum())
urban_pct     = urban_count  / valid_total * 100
rock_pct      = rock_count   / valid_total * 100
natural_pct   = natural_count/ valid_total * 100

print(f"    全域有效像元: {valid_total:,}")
print(f"    ├─ 自然植被区: {natural_count:,} ({natural_pct:.1f}%)  → 动态CR")
print(f"    ├─ 山区裸岩区: {rock_count:,}  ({rock_pct:.1f}%)   → CR={0.70}")
print(f"    └─ 城市建成区: {urban_count:,}  ({urban_pct:.1f}%)   → CR={0.90}")

# 填充城市/裸岩区土壤参数
ks  = np.where(urban_mask, KS_URBAN,  np.where(rock_mask, KS_ROCK,  ks))
ths = np.where(urban_mask, THS_URBAN, np.where(rock_mask, THS_ROCK, ths))
psi_abs = np.where(urban_mask, PSI_URBAN, np.where(rock_mask, PSI_ROCK, psi_abs))

# 统计
ks_stats  = print_stats('Ks(mm/h)',   ks)
ths_stats = print_stats('θs(m³/m³)', ths)
psi_stats = print_stats('|Psi|(cm)', psi_abs)


# ============================================================
# ★ 3.5/4  保存所有静态NPY文件（v5.0新增 acc.npy）
# ============================================================
print("\n[3.5/4] 保存静态NPY文件...")

save_dict = {
    'fdir.npy':         np.array(fdir, dtype=np.float32),
    'slope.npy':        slope_arr,
    'hand.npy':         hand_arr,
    'acc.npy':          acc_arr,          # ★ v5.0新增，Step3 TWI计算需要
    'ks.npy':           ks,
    'ths.npy':          ths,
    'psi.npy':          psi_abs,
    'nodata_mask.npy':  nodata_mask_dem.astype(np.uint8),
    'urban_mask.npy':   urban_mask.astype(np.uint8),
    'rock_mask.npy':    rock_mask.astype(np.uint8),
    'natural_mask.npy': natural_mask.astype(np.uint8),
}

for fname, arr in save_dict.items():
    fpath = os.path.join(OUTPUT_DIR, fname)
    np.save(fpath, arr)
    # 验证
    arr_v = arr[~nodata_mask_dem] if fname != 'nodata_mask.npy' else arr.ravel()
    print(f"    ✅ {fname:25s}  shape={arr.shape}"
          + (f"  max={arr_v.max():.2f}" if arr_v.dtype.kind == 'f'
             else f"  sum={arr_v.sum():,}"))

print(f"\n    共保存 {len(save_dict)} 个NPY文件（含★新增 acc.npy）")


# ============================================================
# 4/4  可视化
# ============================================================
print("\n[4/4] 生成可视化图像...")

# 图1：六图总览
fig = plt.figure(figsize=(18, 12))
fig.patch.set_alpha(0)
fig.suptitle('Step 1 (v5.0)：北京市静态下垫面特征提取总览\n'
             '三类下垫面：城市建成区 | 山区裸岩区 | 自然植被区',
             fontsize=15, fontweight='bold', y=0.99)
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
panels = [
    (fig.add_subplot(gs[0,0]), dem_clean,  'terrain',   '① DEM高程(m)',         dem_stats),
    (fig.add_subplot(gs[0,1]), slope_arr,  'RdYlGn_r',  '② 坡度(°)',            slope_stats),
    (fig.add_subplot(gs[0,2]), hand_arr,   'Blues_r',   '③ HAND(m)',            hand_stats),
    (fig.add_subplot(gs[1,0]), ks,         'YlOrRd',    '④ 饱和导水率Ks(mm/h)', ks_stats),
    (fig.add_subplot(gs[1,1]), ths,        'BrBG',      '⑤ 饱和含水量θs',       ths_stats),
    (fig.add_subplot(gs[1,2]), psi_abs,    'PuBu',      '⑥ 毛管吸力|Psi|(cm)',  psi_stats),
]
for ax, data, cmap, title, st in panels:
    ax.set_facecolor('none'); ax.patch.set_alpha(0)
    sm = imshow_masked(ax, data, nodata_mask_full, cmap)
    ax.set_title(title, fontsize=11, fontweight='bold'); ax.axis('off')
    cb = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, aspect=20)
    ax.text(0.02, 0.02,
            f"均={st['mean']:.3f}\nCV={st['cv']:.3f}",
            transform=ax.transAxes, fontsize=8.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
out1 = os.path.join(VIS_DIR, 'Step1_01_Overview.png')
plt.savefig(out1, dpi=200, bbox_inches='tight', transparent=True); plt.close()
print(f"    [图1] 六图总览  → {out1}")

# 图2：三类下垫面分布
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.patch.set_alpha(0)
fig2.suptitle('Step 1 (v5.0)：三类下垫面空间分布', fontsize=14, fontweight='bold')
ax_cls = axes2[0]
cls_map = np.full((h, w), np.nan)
cls_map[natural_mask] = 0; cls_map[rock_mask] = 1; cls_map[urban_mask] = 2
cmap3 = ListedColormap(['#4CAF50', '#A0522D', '#E91E63'])
ax_cls.imshow(cls_map, cmap=cmap3, vmin=-0.5, vmax=2.5, interpolation='nearest')
ax_cls.set_title('三类下垫面空间分布', fontsize=13, fontweight='bold'); ax_cls.axis('off')
patches_cls = [
    mpatches.Patch(color='#4CAF50', label=f'自然植被区 ({natural_pct:.1f}%)'),
    mpatches.Patch(color='#A0522D', label=f'山区裸岩区 ({rock_pct:.1f}%)'),
    mpatches.Patch(color='#E91E63', label=f'城市建成区 ({urban_pct:.1f}%)'),
]
ax_cls.legend(handles=patches_cls, loc='lower right', fontsize=10, framealpha=0.9)
ax_pie = axes2[1]
wedges, texts, autotexts = ax_pie.pie(
    [natural_count, rock_count, urban_count],
    labels=[f'自然植被区\n{natural_pct:.1f}%',
            f'山区裸岩区\n{rock_pct:.1f}%',
            f'城市建成区\n{urban_pct:.1f}%'],
    colors=['#4CAF50', '#A0522D', '#E91E63'],
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
)
ax_pie.set_title('三类下垫面面积占比', fontsize=12, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(VIS_DIR, 'Step1_02_ThreeClass.png')
plt.savefig(out2, dpi=200, bbox_inches='tight', transparent=True); plt.close()
print(f"    [图2] 三类分布  → {out2}")

# 图3：汇流累积量（含★新增acc验证）
fig3, ax3 = plt.subplots(figsize=(12, 8)); fig3.patch.set_alpha(0)
sm_acc = imshow_masked(ax3, np.log1p(acc_arr), nodata_mask_full, 'Blues')
river_vis = (acc_arr > 500) & ~nodata_mask_full
ax3.imshow(np.where(river_vis, 1.0, np.nan), cmap='Reds', alpha=0.8, vmin=0, vmax=1)
ax3.set_title('汇流累积量（对数）+ 提取水系（红色，>500像元）\n'
              '★ acc.npy 已保存，供Step3 TWI计算',
              fontsize=13, fontweight='bold'); ax3.axis('off')
plt.colorbar(sm_acc, ax=ax3, shrink=0.8, pad=0.02, label='log(acc+1)')
ax3.text(0.02, 0.02,
         f"水系像元: {river_vis.sum():,} ({river_vis.sum()/valid_pixels*100:.2f}%)\n"
         f"acc max={acc_arr[~nodata_mask_full].max():.0f}",
         transform=ax3.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))
plt.tight_layout()
out3 = os.path.join(VIS_DIR, 'Step1_03_Accumulation.png')
plt.savefig(out3, dpi=200, bbox_inches='tight', transparent=True); plt.close()
print(f"    [图3] 汇流+水系 → {out3}")


# ============================================================
# 统计汇总
# ============================================================
print("\n" + "=" * 70)
print("Step 1 (v5.0) 统计汇总")
print("=" * 70)
print(f"  全域有效像元: {valid_total:,}")
print(f"  城市建成区:   {urban_count:,} ({urban_pct:.1f}%)  CR=0.90")
print(f"  山区裸岩区:   {rock_count:,}  ({rock_pct:.1f}%)  CR≈0.70")
print(f"  自然植被区:   {natural_count:,} ({natural_pct:.1f}%)  动态CR")

print(f"\n  ★ v5.0新增输出: acc.npy")
print(f"     → 供 Step3 计算 TWI = ln(acc × cell_area / tan(slope))")

all_stats = [dem_stats, slope_stats, hand_stats, ks_stats, ths_stats, psi_stats]
header = (f"{'指标':<22}{'有效像元':>10}{'Min':>10}{'Max':>10}"
          f"{'Mean':>10}{'Std':>10}{'CV':>8}{'偏度':>8}")
print(f"\n{header}\n" + "-" * 95)
for s in all_stats:
    print(f"{s['name']:<22}{s['count']:>10,}{s['min']:>10.4f}{s['max']:>10.4f}"
          f"{s['mean']:>10.4f}{s['std']:>10.4f}{s['cv']:>8.4f}{s['skew']:>8.4f}")

print(f"\n✅ Step 1 (v5.0) 完成！")
print(f"   NPY文件  → {OUTPUT_DIR}")
print(f"   可视化   → {VIS_DIR}")
print(f"   共 {len(save_dict)} 个NPY：fdir / slope / hand / acc / "
      f"ks / ths / psi / nodata_mask / urban_mask / rock_mask / natural_mask")