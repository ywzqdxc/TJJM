"""
逐年径流系数(CR)与降雨量(R)空间分布演化图
==================================================
特点：
  1. 自动读取 2012-2024 共13年的降雨和CR数据。
  2. ★ 强制使用“全局统一色阶”，保证不同年份颜色的绝对可比性。
  3. 自动排版为 4x4 的论文拼图矩阵（留白处放置全局图例）。
  4. 沿用 v3.0 的精准边界裁剪和纯白背景技术。
  5. 自动导出拼图及逐年高清独立单图。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import rasterio
import os, warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 全局配置
# ============================================================
STATIC_DIR = r'./Step_New/Static'
DYN_DIR = r'./Step_New/Dynamic'
DEM_PATH = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
OUTPUT_DIR = r'./Step_New/Visualization/Annual_Maps'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'Individual'), exist_ok=True)

# 边界控制
CITY_BOUNDARY_SHP = r'E:\Data\src\Beijing\北京市_市.shp'
GRID_DS = 2  # 拼图降采样率（提高出图速度，出最终图可改为1）
INDIVIDUAL_DS = 1  # 单图降采样率（全分辨率）
EXPORT_INDIVIDUAL = True
STUDY_YEARS = list(range(2012, 2025))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ============================================================
# 2. 加载基准网格与边界掩膜
# ============================================================
print("[1/4] 加载基准与边界...")
with rasterio.open(DEM_PATH) as ref:
    h, w, dst_crs, dst_transform, dst_bounds = ref.height, ref.width, ref.crs, ref.transform, ref.bounds
    extent = [dst_bounds.left, dst_bounds.right, dst_bounds.bottom, dst_bounds.top]

valid_mask = ~np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
city_mask = valid_mask.copy()
city_boundary_gdf = None

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
except ImportError:
    print("  ⚠️ Geopandas未安装，跳过轮廓掩膜裁剪。")

# ============================================================
# 3. 数据加载与全局色标计算
# ============================================================
print("[2/4] 加载逐年数据并计算全局色标...")

annual_rain = {}
annual_cr = {}

all_rain_vals = []
all_cr_vals = []

for yr in STUDY_YEARS:
    r_path = os.path.join(DYN_DIR, f'Precipitation_{yr}.npy')
    c_path = os.path.join(DYN_DIR, f'CR_{yr}.npy')

    if os.path.exists(r_path) and os.path.exists(c_path):
        r_arr = np.load(r_path).astype(np.float32)
        c_arr = np.load(c_path).astype(np.float32)

        # 裁剪到城市边界
        r_arr = np.where(city_mask, r_arr, np.nan)
        c_arr = np.where(city_mask, c_arr, np.nan)

        annual_rain[yr] = r_arr
        annual_cr[yr] = c_arr

        all_rain_vals.append(r_arr[np.isfinite(r_arr)])
        all_cr_vals.append(c_arr[np.isfinite(c_arr)])
    else:
        print(f"  ⚠️ {yr} 数据不全，已跳过")

# 计算全局极值 (用于统一色标)
rain_concat = np.concatenate(all_rain_vals)
cr_concat = np.concatenate(all_cr_vals)

# 使用百分位数避免极端异常值拉伸色标
r_vmin, r_vmax = np.percentile(rain_concat, 2), np.percentile(rain_concat, 99)
c_vmin, c_vmax = np.percentile(cr_concat, 1), np.percentile(cr_concat, 99)

print(f"  ✅ 降雨量全局统一色标: [{r_vmin:.1f}, {r_vmax:.1f}] mm")
print(f"  ✅ 径流系数全局统一色标: [{c_vmin:.3f}, {c_vmax:.3f}]")


# ============================================================
# 4. 核心渲染函数
# ============================================================
def render_annual_ax(ax, arr, vmin, vmax, cmap_name, title, is_grid=False, ds=2):
    """单图渲染核心引擎"""
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('white', alpha=0)  # 无数据区域透明

    # 绘图
    im = ax.imshow(arr[::ds, ::ds], cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin='upper', interpolation='nearest', zorder=2)

    # 叠加边界
    if city_boundary_gdf is not None:
        city_boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=1.0, zorder=5)

    # 坐标系清理
    pad_x = (dst_bounds.right - dst_bounds.left) * 0.01
    pad_y = (dst_bounds.top - dst_bounds.bottom) * 0.01
    ax.set_xlim(dst_bounds.left - pad_x, dst_bounds.right + pad_x)
    ax.set_ylim(dst_bounds.bottom - pad_y, dst_bounds.top + pad_y)
    ax.set_xticks([]);
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # 如果是独立单图，自带图例、比例尺、指北针
    if not is_grid:
        NX, NY = 0.88, 0.92
        ax.annotate('', xy=(NX, NY + 0.07), xytext=(NX, NY), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=14))
        ax.text(NX, NY + 0.10, 'N', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        # 比例尺
        bar_frac = 30 / (111.32 * np.cos(np.radians((dst_bounds.top + dst_bounds.bottom) / 2))) / (
                    dst_bounds.right - dst_bounds.left)
        bar_x, bar_y = 0.05, 0.03
        ax.plot([bar_x, bar_x + bar_frac], [bar_y, bar_y], transform=ax.transAxes, color='black', lw=2, zorder=10)
        for bx in [bar_x, bar_x + bar_frac]:
            ax.plot([bx, bx], [bar_y - 0.01, bar_y + 0.01], transform=ax.transAxes, color='black', lw=1.5, zorder=10)
        ax.text(bar_x + bar_frac / 2, bar_y - 0.025, '0        30 km', ha='center', va='top', fontsize=7,
                transform=ax.transAxes)

    ax.set_title(title, fontsize=12 if not is_grid else 10, pad=8, loc='center')
    return im


def generate_4x4_grid(data_dict, vmin, vmax, cmap, title_prefix, out_filename):
    """生成包含13年的 4x4 拼图"""
    fig = plt.figure(figsize=(16, 15), facecolor='white')
    fig.suptitle(f'{title_prefix} (2012-2024 全局统一色阶)', fontsize=18, fontweight='bold', y=0.96)

    gs = GridSpec(4, 4, figure=fig, left=0.02, right=0.98, top=0.92, bottom=0.05, hspace=0.15, wspace=0.05)

    # 绘制前13个图
    for idx, yr in enumerate(STUDY_YEARS):
        row, col = idx // 4, idx % 4
        ax = fig.add_subplot(gs[row, col])
        arr = data_dict[yr]

        im = render_annual_ax(ax, arr, vmin, vmax, cmap, f'{yr}年', is_grid=True, ds=GRID_DS)

        # 仅第一幅图右上角加指北针
        if idx == 0:
            NX, NY = 0.85, 0.88
            ax.annotate('', xy=(NX, NY + 0.1), xytext=(NX, NY), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=12))
            ax.text(NX, NY + 0.13, 'N', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')

    # 在剩下的空白区域(第14,15,16个格子)放置全局图例和比例尺
    cbar_ax = fig.add_subplot(gs[3, 1:3])
    cbar_ax.axis('off')

    cbar = plt.colorbar(im, ax=cbar_ax, orientation='horizontal', fraction=0.8, pad=0.1)
    cbar.set_label(f'{title_prefix}值', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # 添加大比例尺
    bar_frac = 50 / (111.32 * np.cos(np.radians((dst_bounds.top + dst_bounds.bottom) / 2))) / (
                dst_bounds.right - dst_bounds.left)
    bx0, bx1 = 0.2, 0.2 + bar_frac
    by = 0.1
    cbar_ax.plot([bx0, bx1], [by, by], transform=cbar_ax.transAxes, color='black', lw=2.5)
    for bx in [bx0, bx1]:
        cbar_ax.plot([bx, bx], [by - 0.05, by + 0.05], transform=cbar_ax.transAxes, color='black', lw=2)
    cbar_ax.text((bx0 + bx1) / 2, by - 0.15, '0           50 km', ha='center', va='top', fontsize=10,
                 transform=cbar_ax.transAxes)

    out_path = os.path.join(OUTPUT_DIR, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 拼图保存成功: {out_filename}")


# ============================================================
# 5. 执行绘制
# ============================================================
print("\n[3/4] 正在生成 4x4 时空演变拼图...")

# 降雨量拼图 (Blues)
generate_4x4_grid(annual_rain, r_vmin, r_vmax, 'Blues', '北京市逐年汛期降雨量空间分布',
                  'Precipitation_13Years_Grid.png')

# 径流系数CR拼图 (YlOrRd - 红橙色反映不透水面增加)
generate_4x4_grid(annual_cr, c_vmin, c_vmax, 'YlOrRd', '北京市逐年地表径流系数(CR)空间分布', 'CR_13Years_Grid.png')

# ============================================================
# 6. 单图导出 (用于PPT或单张大图排版)
# ============================================================
if EXPORT_INDIVIDUAL:
    print("\n[4/4] 正在导出高清独立单图...")
    for yr in STUDY_YEARS:
        # 降雨单图
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        render_annual_ax(ax, annual_rain[yr], r_vmin, r_vmax, 'Blues', f'北京市汛期降雨量 ({yr})', is_grid=False,
                         ds=INDIVIDUAL_DS)
        fig.savefig(os.path.join(OUTPUT_DIR, 'Individual', f'Rain_{yr}.png'), bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # CR单图
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        render_annual_ax(ax, annual_cr[yr], c_vmin, c_vmax, 'YlOrRd', f'北京市地表径流系数 CR ({yr})', is_grid=False,
                         ds=INDIVIDUAL_DS)
        fig.savefig(os.path.join(OUTPUT_DIR, 'Individual', f'CR_{yr}.png'), bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f"  ✅ 26 张高清独立单图已保存至: {os.path.join(OUTPUT_DIR, 'Individual')}")

print("\n🎉 逐年空间演化制图全部完成！")