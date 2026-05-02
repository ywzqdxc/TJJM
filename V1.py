"""
指标空间分布图 v3.0 (终极重构版)
================================
优化：
  1. 彻底遵循 DRY 原则，提炼核心绘图函数，消除所有冗余重复代码。
  2. 将第 10 个指标（径流系数 CR）无缝整合进全局配置。
  3. 统一控制降采样参数，一键切换“测试/高画质”模式。
  4. 支持自动生成 3x3 论文主图 + 10 张独立高清附图。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.warp import reproject, Resampling
import os, warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 全局配置
# ============================================================
STATIC_DIR = r'./Step_New/Static'
DYN_DIR = r'./Step_New/Dynamic'
EXT_DIR = r'./Step_New/External'
DEM_PATH = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
OUTPUT_DIR = r'./Step_New/Visualization/Indicator_Maps'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'Individual'), exist_ok=True)

# 边界与渲染控制
BOUNDARY_SHP = None  # 区县级边界 (若无置为 None)
CITY_BOUNDARY_SHP = r'E:\Data\src\Beijing\北京市_市.shp'  # 全市轮廓
GRID_DS = 1  # 3x3 拼图的降采样率（1=全分辨率，测试可改3）
INDIVIDUAL_DS = 1  # 单图导出的降采样率（1=全分辨率）
EXPORT_INDIVIDUAL = True  # 是否导出10张独立单图

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# 指标全家桶 (10个)
INDICATORS = [
    dict(label='(a) 汛期降雨量', file='Precipitation_Mean.npy', dir=DYN_DIR, is_tif=False, cmap='Blues', n=5),
    dict(label='(b) 地形湿度指数 TWI', file='twi.npy', dir=STATIC_DIR, is_tif=False, cmap='GnBu', n=5),
    dict(label='(c) 低洼度 HAND', file='hand.npy', dir=STATIC_DIR, is_tif=False, cmap='YlOrBr', n=5),
    dict(label='(d) 历史积水点密度', file='waterlogging_point_density_30m.tif', dir=EXT_DIR, is_tif=True,
         cmap='Purples', n=5),
    dict(label='(e) 人口密度', file='population_density_30m.tif', dir=EXT_DIR, is_tif=True, cmap='RdPu', n=5),
    dict(label='(f) 道路网络密度', file='road_density_30m.tif', dir=EXT_DIR, is_tif=True, cmap='YlGn', n=5),
    dict(label='(g) 应急避难场所密度', file='shelter_density_30m.tif', dir=EXT_DIR, is_tif=True, cmap='copper', n=5),
    dict(label='(h) 综合医院密度', file='hospital_density_30m.tif', dir=EXT_DIR, is_tif=True, cmap='PuBu', n=5),
    dict(label='(i) 消防救援站密度', file='firestation_density_30m.tif', dir=EXT_DIR, is_tif=True, cmap='BuPu', n=5),
    dict(label='(j) 径流系数 CR', file='CR_MultiYear_Mean.npy', dir=DYN_DIR, is_tif=False, cmap='YlOrRd', n=5)
]


# ============================================================
# 2. 核心通用函数
# ============================================================

def fmt(v):
    """数值智能格式化"""
    if abs(v) >= 100:  return f'{v:.0f}'
    if abs(v) >= 10:   return f'{v:.1f}'
    if abs(v) >= 1:    return f'{v:.2f}'
    if abs(v) >= 0.01: return f'{v:.3f}'
    return f'{v:.4f}'


def get_breaks(arr, city_mask, n=5):
    """计算分位数断点"""
    vals = arr[city_mask & np.isfinite(arr)]
    if vals.size < n: return np.linspace(0, 1, n + 1)
    bk = np.unique(np.percentile(vals, np.linspace(0, 100, n + 1)))
    if len(bk) < n + 1: bk = np.linspace(vals.min(), vals.max(), n + 1)
    bk[-1] *= 1.0001
    return bk


def render_map_on_ax(ax, arr, cfg, ds, is_grid=False):
    """核心绘图引擎：在指定的 ax 上渲染地图、图例、边界、比例尺等一切元素"""
    ax.set_aspect('equal');
    ax.set_facecolor('white')

    if arr is None:
        ax.text(0.5, 0.5, '数据缺失', ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_title(cfg['label'], fontsize=10, pad=6);
        ax.axis('off')
        return

    # 1. 颜色与断点
    n = cfg['n']
    bks = get_breaks(arr, city_mask, n=n)
    base_cmap = plt.get_cmap(cfg['cmap'])
    colors = [base_cmap(0.15 + 0.70 * i / (n - 1)) for i in range(n)]
    cmap_d = mcolors.ListedColormap(colors)
    norm_d = mcolors.BoundaryNorm(bks, ncolors=n)
    cmap_d.set_bad(color='white', alpha=0)  # 背景透明

    # 2. 绘制栅格
    ax.imshow(arr[::ds, ::ds], cmap=cmap_d, norm=norm_d, extent=extent,
              origin='upper', interpolation='nearest', zorder=2)

    # 3. 叠加边界
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.8, zorder=4)
    if city_boundary_gdf is not None:
        city_boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0, alpha=1.0, zorder=5)

    # 4. 设置范围与清理坐标轴
    pad_x, pad_y = (dst_bounds.right - dst_bounds.left) * 0.01, (dst_bounds.top - dst_bounds.bottom) * 0.01
    ax.set_xlim(dst_bounds.left - pad_x, dst_bounds.right + pad_x)
    ax.set_ylim(dst_bounds.bottom - pad_y, dst_bounds.top + pad_y)
    ax.set_xticks([]);
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # 5. 绘制图例
    LX, LY, BW, BH, GAP = 0.02, 0.97, 0.04, 0.055, 0.004
    ax.text(LX, LY + 0.012, '指标归一化值', transform=ax.transAxes, fontsize=6.5, fontweight='bold',
            va='bottom', ha='left', path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    for i, color in enumerate(colors):
        y_top, y_bot = LY - i * (BH + GAP), LY - i * (BH + GAP) - BH
        ax.add_patch(plt.Rectangle((LX, y_bot), BW, BH, facecolor=color, edgecolor='#AAAAAA',
                                   linewidth=0.4, transform=ax.transAxes, zorder=8))
        ax.text(LX + BW + 0.012, (y_top + y_bot) / 2, f'{fmt(bks[i])}~{fmt(bks[i + 1])}',
                transform=ax.transAxes, fontsize=6.0, va='center', ha='left',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # 6. 绘制指北针 (如果是单图，或者拼图的右上角)
    if not is_grid or getattr(ax, 'is_top_right', False):
        NX, NY = 0.88, 0.92
        ax.annotate('', xy=(NX, NY + 0.07), xytext=(NX, NY), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=14))
        ax.text(NX, NY + 0.10, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 7. 绘制内部比例尺 (如果是单图)
    if not is_grid:
        bar_frac = 30 / (111.32 * np.cos(np.radians((dst_bounds.top + dst_bounds.bottom) / 2))) / (
                    dst_bounds.right - dst_bounds.left)
        bar_x, bar_y = 0.05, 0.03
        ax.plot([bar_x, bar_x + bar_frac], [bar_y, bar_y], transform=ax.transAxes, color='black', lw=2, zorder=10)
        for bx in [bar_x, bar_x + bar_frac]:
            ax.plot([bx, bx], [bar_y - 0.01, bar_y + 0.01], transform=ax.transAxes, color='black', lw=1.5, zorder=10)
        ax.text(bar_x + bar_frac / 2, bar_y - 0.025, '0        30 km', ha='center', va='top', fontsize=7,
                transform=ax.transAxes)

    ax.set_title(cfg['label'], fontsize=11 if not is_grid else 9.5, pad=10 if not is_grid else 5, loc='center')


# ============================================================
# 3. 加载基准网格与边界掩膜
# ============================================================
print("[1/3] 加载基准与边界...")
with rasterio.open(DEM_PATH) as ref:
    h, w, dst_crs, dst_transform, dst_bounds = ref.height, ref.width, ref.crs, ref.transform, ref.bounds
    extent = [dst_bounds.left, dst_bounds.right, dst_bounds.bottom, dst_bounds.top]

valid_mask = ~np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
city_mask = valid_mask.copy()
boundary_gdf, city_boundary_gdf = None, None

try:
    import geopandas as gpd
    from shapely.geometry import mapping
    from shapely.ops import unary_union
    from rasterio.features import geometry_mask as geo_mask

    if os.path.exists(CITY_BOUNDARY_SHP):
        city_boundary_gdf = gpd.read_file(CITY_BOUNDARY_SHP).to_crs(dst_crs.to_epsg())
        city_mask = geo_mask([mapping(unary_union(city_boundary_gdf.geometry))],
                             transform=dst_transform, out_shape=(h, w), invert=True) & valid_mask
        print(f"  ✅ 城市掩膜生成: {city_mask.sum():,} 个有效像元")

    if BOUNDARY_SHP and os.path.exists(BOUNDARY_SHP):
        boundary_gdf = gpd.read_file(BOUNDARY_SHP).to_crs(dst_crs.to_epsg())
except ImportError:
    print("  ⚠️ Geopandas未安装，跳过轮廓掩膜裁剪。")

# ============================================================
# 4. 加载所有数据矩阵
# ============================================================
print("[2/3] 加载指标数据...")
arrays = []
for cfg in INDICATORS:
    path = os.path.join(cfg['dir'], cfg['file'])
    if not os.path.exists(path):
        arrays.append(None);
        print(f"  ❌ 缺失: {cfg['label']}");
        continue

    if not cfg['is_tif']:
        arr = np.load(path).astype(np.float32)
    else:
        with rasterio.open(path) as src:
            if src.height == h and src.width == w:
                # Use np.where instead of src.where
                arr = np.where(src.read(1) == src.nodata, np.nan, src.read(1)).astype(np.float32)
            else:
                arr = np.full((h, w), np.nan, dtype=np.float32)
                reproject(source=rasterio.band(src, 1), destination=arr, src_transform=src.transform,
                          src_crs=src.crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.bilinear)

    arr = np.where((arr < -1e10) | ~city_mask, np.nan, arr)
    arrays.append(arr)
    print(f"  ✅ 已加载: {cfg['label']}")

# ============================================================
# 5. 绘制 3x3 拼图主图 (仅取前9个指标)
# ============================================================
print("\n[3/3] 绘制 3x3 论文主拼图...")
fig_grid = plt.figure(figsize=(16, 13), facecolor='white')
gs = GridSpec(3, 3, figure=fig_grid, left=0.01, right=0.99, top=0.97, bottom=0.04, hspace=0.12, wspace=0.04)

for idx in range(9):
    row, col = idx // 3, idx % 3
    ax = fig_grid.add_subplot(gs[row, col])
    ax.is_top_right = (row == 0 and col == 2)  # 标记右上角用于画指北针
    render_map_on_ax(ax, arrays[idx], INDICATORS[idx], ds=GRID_DS, is_grid=True)

    # 在底层添加外部比例尺
    if row == 2:
        bar_frac = 30 / (111.32 * np.cos(np.radians((dst_bounds.top + dst_bounds.bottom) / 2))) / (
                    dst_bounds.right - dst_bounds.left)
        pos = ax.get_position()
        bx0, bx1, by = pos.x0 + 0.01 * (pos.x1 - pos.x0), pos.x0 + 0.01 * (pos.x1 - pos.x0) + bar_frac * (
                    pos.x1 - pos.x0), pos.y0 - 0.022
        fig_grid.add_artist(plt.Line2D([bx0, bx1], [by, by], transform=fig_grid.transFigure, color='black', lw=2))
        for bx in [bx0, bx1]: fig_grid.add_artist(
            plt.Line2D([bx, bx], [by - 0.004, by + 0.004], transform=fig_grid.transFigure, color='black', lw=1.5))
        fig_grid.text((bx0 + bx1) / 2, by - 0.010, '0        30 km', ha='center', va='top', fontsize=7,
                      transform=fig_grid.transFigure)

out_grid = os.path.join(OUTPUT_DIR, 'Indicator_Maps_v3.png')
fig_grid.savefig(out_grid, dpi=300, bbox_inches='tight')
plt.close(fig_grid)
print(f"  ✅ 3x3 拼图保存完毕: {os.path.basename(out_grid)}")

# ============================================================
# 6. 独立导出 10 张高清单图
# ============================================================
if EXPORT_INDIVIDUAL:
    print("\n[独立导出] 开始渲染 10 张独立高清单图...")
    for idx, cfg in enumerate(INDICATORS):
        if arrays[idx] is None: continue

        fig_s = plt.figure(figsize=(6, 6), dpi=300, facecolor='white')
        render_map_on_ax(fig_s.add_subplot(111), arrays[idx], cfg, ds=INDIVIDUAL_DS, is_grid=False)

        clean_name = cfg['label'].replace('(', '').replace(')', '').replace(' ', '_')
        out_single = os.path.join(OUTPUT_DIR, 'Individual', f'{idx + 1:02d}_{clean_name}.png')
        fig_s.savefig(out_single, dpi=300, bbox_inches='tight')
        plt.close(fig_s)
        print(f"  ✅ 保存单图: {os.path.basename(out_single)}")

print("\n🎉 所有绘图任务圆满完成！")