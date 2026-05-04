"""
V4.py: 数据预处理过程学术可视化
================================
对应论文"数据预处理"章节各小节，逐条生成出版级图件：

  Fig1  气象水文 — 降雨量空间分布 + 逐年时序柱状图（双轴）
  Fig2  气象水文 — 径流系数CR空间分布 + 三类下垫面占比
  Fig3  地形水文 — DEM / TWI / HAND / 汇流累积量四图联排
  Fig4  历史灾情 — 积水点核密度 + 积水点位散点叠加
  Fig5  人口密度 — 多年均值空间分布 + 逐年均值折线
  Fig6  应急设施 — 避难场所 / 医院 / 消防站三图联排
  Fig7  数据对齐 — 有效像元掩码 + 10指标数据完整性矩阵热力图

输入：全部来自已有流程输出，不重复计算原始数据
  ./Step_New/Static/         nodata_mask / twi / hand / acc / slope
  ./Step_New/Dynamic/        Precipitation_Mean / CR_MultiYear_Mean
                             Precipitation_{year} / CR_{year}
  ./Step_New/External/       population_density_30m / waterlogging_point_density_30m
                             hospital/shelter/firestation_density_30m
  ./Step_New/Risk_Map/       Vuln_TOPSIS_{year}
  E:/Data/src/DEM数据/       北京市_DEM_30m分辨率_NASA数据.tif
  E:/Data/src/Beijing/       北京市_市.shp / 北京市_区.shp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
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
CITY_SHP   = r'E:\Data\src\Beijing\北京市_市.shp'
DIST_SHP   = r'E:\Data\src\Beijing\北京市_区.shp'
OUT_DIR    = r'./Step_New/Visualization/V4_Preprocessing'
os.makedirs(OUT_DIR, exist_ok=True)

STUDY_YEARS = list(range(2012, 2025))
DS = 4   # 全局降采样率（4=约1/16数据量，保证出图速度）

print("=" * 70)
print("V4.py: 数据预处理学术可视化")
print("=" * 70)

# ============================================================
# 工具函数
# ============================================================
def load_npy(path):
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)

def load_tif(path):
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
    return np.where(arr < -1e10, np.nan, arr)

def ds_arr(arr, mask=None):
    """降采样 + 可选掩膜"""
    a = arr[::DS, ::DS].copy()
    if mask is not None:
        m = mask[::DS, ::DS]
        a = np.where(m & np.isfinite(a), a, np.nan)
    return a

def masked_cmap(name):
    cm = plt.get_cmap(name).copy()
    cm.set_bad('white', alpha=0)
    return cm

# ---- 地图装饰 ----
def _clean_ax(ax, bounds, pad=0.01):
    px = (bounds.right - bounds.left) * pad
    py = (bounds.top - bounds.bottom) * pad
    ax.set_xlim(bounds.left - px, bounds.right + px)
    ax.set_ylim(bounds.bottom - py, bounds.top + py)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor('white')

from matplotlib.lines import Line2D

def add_circle_legend(ax, colors, labels, title, loc='upper left'):
    """仿 Step 3 的圆形离散图例"""
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=c, markersize=10,
                      markeredgecolor='black', markeredgewidth=0.5)
               for c in colors]
    ax.legend(handles, labels, loc=loc, title=title,
              fontsize=8, title_fontsize=9, framealpha=0.8, edgecolor='#CCCCCC')

def add_north(ax, x=0.92, y=0.86, size=0.07, fs=11):
    ax.annotate('', xy=(x, y + size), xytext=(x, y), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0, mutation_scale=14))
    ax.text(x, y + size + 0.02, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=fs, fontweight='bold')

def add_scale(ax, bounds, bar_km=30, bx0=0.60, by=0.05, fs=9):
    center_lat = (bounds.top + bounds.bottom) / 2
    total_x = bounds.right - bounds.left
    bar_deg = bar_km / (111.32 * np.cos(np.radians(center_lat)))
    bar_frac = bar_deg / total_x
    bx1 = bx0 + bar_frac
    ax.plot([bx0, bx1], [by, by], transform=ax.transAxes, color='black', lw=3, zorder=10)
    for bx in [bx0, bx1]:
        ax.plot([bx, bx], [by - 0.012, by + 0.012], transform=ax.transAxes,
                color='black', lw=2, zorder=10)
    ax.text(bx0, by - 0.03, '0', ha='center', va='top',
            transform=ax.transAxes, fontsize=fs)
    ax.text(bx1, by - 0.03, f'{bar_km} km', ha='center', va='top',
            transform=ax.transAxes, fontsize=fs)

def add_boundaries(ax, city_gdf, dist_gdf):
    if dist_gdf is not None:
        dist_gdf.boundary.plot(ax=ax, color='white', linewidth=0.6,
                               linestyle='--', alpha=0.8, zorder=4)
    if city_gdf is not None:
        city_gdf.boundary.plot(ax=ax, color='#333333', linewidth=1.2, zorder=5)

# ============================================================
# 加载基础信息
# ============================================================
print("\n[基础] 加载DEM基准与边界...")
with rasterio.open(DEM_PATH) as ref:
    h, w      = ref.height, ref.width
    bounds    = ref.bounds
    transform = ref.transform
    crs       = ref.crs

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

nodata_mask = load_npy(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

try:
    import geopandas as gpd
    city_gdf = gpd.read_file(CITY_SHP) if os.path.exists(CITY_SHP) else None
    dist_gdf = gpd.read_file(DIST_SHP) if os.path.exists(DIST_SHP) else None
    print("  ✅ 行政边界加载成功")
except Exception as e:
    city_gdf = dist_gdf = None
    print(f"  ⚠️  边界加载失败: {e}")

# ============================================================
# Fig1  降雨量：空间分布 + 逐年时序
# ============================================================
print("\n[Fig1] 降雨量空间分布 + 逐年时序...")

rain_mean = load_npy(os.path.join(DYN_DIR, 'Precipitation_Mean.npy'))

# 逐年均值时序
rain_ts = []
for yr in STUDY_YEARS:
    fp = os.path.join(DYN_DIR, f'Precipitation_{yr}.npy')
    arr = load_npy(fp)
    if arr is not None:
        v = arr[valid_mask & np.isfinite(arr)]
        rain_ts.append(float(v.mean()) if v.size > 0 else np.nan)
    else:
        rain_ts.append(np.nan)
rain_ts = np.array(rain_ts)

fig1 = plt.figure(figsize=(16, 7), facecolor='white')
gs1 = GridSpec(1, 2, figure=fig1, wspace=0.12,
               left=0.04, right=0.96, top=0.88, bottom=0.10)
fig1.suptitle('（一）气象水文数据预处理：汛期降雨量\n'
              '（CHM_PREV2, 0.1°→30m双线性插值，2012—2024年均值）',
              fontsize=13, fontweight='bold')

# 左：空间分布图
ax1l = fig1.add_subplot(gs1[0, 0])
if city_gdf is not None:
    city_gdf.plot(ax=ax1l, color='#F5F5F5', zorder=1)
add_boundaries(ax1l, city_gdf, dist_gdf)
if rain_mean is not None:
    rm_show = ds_arr(rain_mean, valid_mask)
    im = ax1l.imshow(rm_show, cmap=masked_cmap('Blues'), extent=extent,
                     vmin=np.nanpercentile(rm_show, 2),
                     vmax=np.nanpercentile(rm_show, 98), zorder=3,
                     interpolation='nearest')
    cb = plt.colorbar(im, ax=ax1l, shrink=0.75, pad=0.03)
    cb.set_label('汛期累积降雨量 (mm)', fontsize=10)
    cb.outline.set_linewidth(0.6)
_clean_ax(ax1l, bounds)
add_north(ax1l)
add_scale(ax1l, bounds)
ax1l.set_title('(a) 多年平均汛期降雨量空间分布', fontsize=11, fontweight='bold', pad=8)

# 右：逐年柱状 + 趋势线
ax1r = fig1.add_subplot(gs1[0, 1])
years_arr = np.array(STUDY_YEARS)
valid_rain = ~np.isnan(rain_ts)
bars = ax1r.bar(years_arr[valid_rain], rain_ts[valid_rain],
                color='#4292C6', alpha=0.8, width=0.6,
                edgecolor='white', linewidth=0.5, label='逐年汛期降雨量均值')
if valid_rain.sum() >= 3:
    z = np.polyfit(years_arr[valid_rain], rain_ts[valid_rain], 1)
    p = np.poly1d(z)
    ax1r.plot(years_arr, p(years_arr), color='#D7301F', lw=2, linestyle='--',
              label=f'线性趋势 ({z[0]:+.1f} mm/年)', zorder=5)

# 均值参考线
mean_val = float(np.nanmean(rain_ts))
ax1r.axhline(mean_val, color='#636363', lw=1.2, linestyle=':', alpha=0.8,
             label=f'多年均值 {mean_val:.0f} mm')

ax1r.set_xlabel('年份', fontsize=11)
ax1r.set_ylabel('全域均值降雨量 (mm)', fontsize=11)
ax1r.set_xticks(years_arr)
ax1r.set_xticklabels([str(y) for y in STUDY_YEARS], rotation=35, fontsize=8.5)
ax1r.legend(fontsize=9, framealpha=0.9, loc='upper left')
ax1r.grid(axis='y', alpha=0.3, linestyle='--')
ax1r.set_title('(b) 逐年汛期降雨量均值时序', fontsize=11, fontweight='bold', pad=8)
ax1r.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

# 数值标注（偶数年）
for yr, val in zip(years_arr[valid_rain], rain_ts[valid_rain]):
    if yr % 2 == 0:
        ax1r.text(yr, val + 5, f'{val:.0f}', ha='center', va='bottom',
                  fontsize=7, color='#2171B5')

fig1.savefig(os.path.join(OUT_DIR, 'Fig1_Precipitation.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig1_Precipitation.png")


# ============================================================
# Fig2  径流系数CR：空间分布 + 三类下垫面占比
# ============================================================
print("[Fig2] 径流系数CR空间分布 + 下垫面组成...")

cr_mean    = load_npy(os.path.join(DYN_DIR, 'CR_MultiYear_Mean.npy'))
urban_mask = load_npy(os.path.join(STATIC_DIR, 'urban_mask.npy'))
rock_mask  = load_npy(os.path.join(STATIC_DIR, 'rock_mask.npy'))
nat_mask   = load_npy(os.path.join(STATIC_DIR, 'natural_mask.npy'))

fig2 = plt.figure(figsize=(16, 7), facecolor='white')
gs2 = GridSpec(1, 2, figure=fig2, wspace=0.14,
               left=0.04, right=0.96, top=0.88, bottom=0.10)
fig2.suptitle('（一）气象水文数据预处理：地表径流系数（SCS-CN模型）\n'
              '三类下垫面：城市建成区（CR=0.90）/ 山区裸岩（CR≈0.70）/ 自然植被（动态）',
              fontsize=13, fontweight='bold')

# 左：CR空间分布
ax2l = fig2.add_subplot(gs2[0, 0])
if city_gdf is not None:
    city_gdf.plot(ax=ax2l, color='#F5F5F5', zorder=1)
add_boundaries(ax2l, city_gdf, dist_gdf)

if cr_mean is not None:
    cr_show = ds_arr(cr_mean, valid_mask)
    im2 = ax2l.imshow(cr_show, cmap=masked_cmap('YlOrRd'), extent=extent,
                      vmin=np.nanpercentile(cr_show, 2),
                      vmax=np.nanpercentile(cr_show, 98), zorder=3,
                      interpolation='nearest')
    cb2 = plt.colorbar(im2, ax=ax2l, shrink=0.75, pad=0.03)
    cb2.set_label('地表径流系数 CR', fontsize=10)
    cb2.outline.set_linewidth(0.6)

_clean_ax(ax2l, bounds)
add_north(ax2l)
add_scale(ax2l, bounds)
ax2l.set_title('(a) 多年平均地表径流系数空间分布', fontsize=11, fontweight='bold', pad=8)

# 右：三类下垫面空间图 + 饼图嵌入
ax2r = fig2.add_subplot(gs2[0, 1])
if city_gdf is not None:
    city_gdf.plot(ax=ax2r, color='#F5F5F5', zorder=1)
add_boundaries(ax2r, city_gdf, dist_gdf)

if urban_mask is not None and rock_mask is not None and nat_mask is not None:
    cls_map = np.full((h, w), np.nan, dtype=np.float32)
    cls_map[nat_mask.astype(bool) & valid_mask]   = 0
    cls_map[rock_mask.astype(bool) & valid_mask]  = 1
    cls_map[urban_mask.astype(bool) & valid_mask] = 2

    cmap_cls = ListedColormap(['#74C476', '#A0522D', '#E6550D'])
    cmap_cls.set_bad('white', alpha=0)
    norm_cls = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

    cls_show = cls_map[::DS, ::DS]
    ax2r.imshow(cls_show, cmap=cmap_cls, norm=norm_cls, extent=extent,
                zorder=3, interpolation='nearest')

    n_nat   = int(nat_mask.astype(bool).sum())
    n_rock  = int(rock_mask.astype(bool).sum())
    n_urban = int(urban_mask.astype(bool).sum())
    n_total = n_nat + n_rock + n_urban if (n_nat + n_rock + n_urban) > 0 else 1

    patches2 = [
        mpatches.Patch(color='#74C476',
                       label=f'自然植被区 ({n_nat/n_total*100:.1f}%)'),
        mpatches.Patch(color='#A0522D',
                       label=f'山区裸岩区 ({n_rock/n_total*100:.1f}%)'),
        mpatches.Patch(color='#E6550D',
                       label=f'城市建成区 ({n_urban/n_total*100:.1f}%)'),
    ]
    ax2r.legend(handles=patches2, loc='lower left', fontsize=9.5,
                framealpha=0.93, edgecolor='#CCCCCC')

    # 内嵌饼图（右下角）
    ax_pie = ax2r.inset_axes([0.65, 0.03, 0.33, 0.33])
    wedges, _, autotexts = ax_pie.pie(
        [n_nat, n_rock, n_urban],
        colors=['#74C476', '#A0522D', '#E6550D'],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'},
        pctdistance=0.72
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax_pie.set_facecolor('white')

_clean_ax(ax2r, bounds)
add_north(ax2r)
add_scale(ax2r, bounds)
ax2r.set_title('(b) 三类下垫面空间分布', fontsize=11, fontweight='bold', pad=8)

fig2.savefig(os.path.join(OUT_DIR, 'Fig2_Runoff_LandCover.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig2_Runoff_LandCover.png")


# ============================================================
# Fig3  地形水文四图联排：DEM / 坡度 / TWI / HAND
# ============================================================
print("[Fig3] 地形水文四图联排...")

dem_arr   = load_tif(DEM_PATH)
slope_arr = load_npy(os.path.join(STATIC_DIR, 'slope.npy'))
twi_arr   = load_npy(os.path.join(STATIC_DIR, 'twi.npy'))
hand_arr  = load_npy(os.path.join(STATIC_DIR, 'hand.npy'))
acc_arr   = load_npy(os.path.join(STATIC_DIR, 'acc.npy'))

panels3 = [
    (dem_arr,   'terrain',    'DEM高程 (m)',
     '(a) 数字高程模型（SRTM 30m）'),
    (slope_arr, 'RdYlGn_r',  '坡度 (°)',
     '(b) 地面坡度（Sobel算子）'),
    (twi_arr,   'GnBu',      'TWI = ln(α/tanβ)',
     '(c) 地形湿度指数（TWI）'),
    (hand_arr,  'Blues_r',   'HAND (m)',
     '(d) 相对高程（HAND）'),
]

fig3, axes3 = plt.subplots(1, 4, figsize=(22, 7), facecolor='white')
fig3.suptitle('（一）地形水文参数预处理：基于30m SRTM DEM的pysheds提取结果',
              fontsize=13, fontweight='bold', y=1.01)

for ax, (data, cmap_name, cbar_label, title) in zip(axes3, panels3):
    if city_gdf is not None:
        city_gdf.plot(ax=ax, color='#F0F0F0', zorder=1)
    add_boundaries(ax, city_gdf, dist_gdf)

    if data is not None:
        d_show = ds_arr(data, valid_mask)
        vlo = np.nanpercentile(d_show, 2)
        vhi = np.nanpercentile(d_show, 98)
        im3 = ax.imshow(d_show, cmap=masked_cmap(cmap_name), extent=extent,
                        vmin=vlo, vmax=vhi, zorder=3, interpolation='nearest')
        cb3 = plt.colorbar(im3, ax=ax, shrink=0.75, pad=0.03, orientation='horizontal')
        cb3.set_label(cbar_label, fontsize=9)
        cb3.outline.set_linewidth(0.6)
        cb3.ax.tick_params(labelsize=8)

        # 统计标注
        v = data[valid_mask & np.isfinite(data)]
        ax.text(0.03, 0.97,
                f'均值: {v.mean():.2f}\n中位: {np.median(v):.2f}',
                transform=ax.transAxes, fontsize=8.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor='#AAAAAA'))
    _clean_ax(ax, bounds)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)

# 仅第一张加指北针和比例尺
add_north(axes3[0], x=0.88, y=0.84)
add_scale(axes3[0], bounds, bar_km=30, bx0=0.55, by=0.05)

# 叠加水系（在HAND子图上）
if acc_arr is not None:
    river = (acc_arr > 500) & valid_mask
    rv_show = np.where(river[::DS, ::DS], 1.0, np.nan)
    cmap_rv = plt.get_cmap('Reds').copy()
    cmap_rv.set_bad('white', alpha=0)
    axes3[3].imshow(rv_show, cmap=cmap_rv, alpha=0.6, extent=extent,
                    vmin=0, vmax=1, zorder=4, interpolation='nearest')

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'Fig3_Terrain_Hydrology.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig3_Terrain_Hydrology.png")


# ============================================================
# Fig4  历史积水点：核密度空间分布 + 散点位置图
# ============================================================
print("[Fig4] 历史积水点核密度...")

wl_density = load_tif(os.path.join(EXT_DIR, 'waterlogging_point_density_30m.tif'))

# 尝试加载原始积水点CSV（供散点叠加）
wl_csv = os.path.join(EXT_DIR, 'waterlog_all_points.csv')
wl_pts = None
if os.path.exists(wl_csv):
    try:
        wl_pts = pd.read_csv(wl_csv)
        wl_pts = wl_pts[(wl_pts['lon'] >= bounds.left) & (wl_pts['lon'] <= bounds.right) &
                        (wl_pts['lat'] >= bounds.bottom) & (wl_pts['lat'] <= bounds.top)]
    except Exception:
        wl_pts = None

fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
fig4.suptitle('（二）历史灾情数据预处理：北京市积水点高斯核密度估计（2012—2024）\n'
              '（Silverman带宽法，σ≈500m，输出30m栅格）',
              fontsize=13, fontweight='bold')

for ax in axes4:
    if city_gdf is not None:
        city_gdf.plot(ax=ax, color='#F5F5F5', zorder=1)
    add_boundaries(ax, city_gdf, dist_gdf)

# 左：核密度连续面
ax4l = axes4[0]
if wl_density is not None:
    wd_show = ds_arr(wl_density, valid_mask)
    wd_show_masked = np.ma.masked_where(~np.isfinite(wd_show) | (wd_show < 0.001), wd_show)
    im4 = ax4l.imshow(wd_show_masked, cmap=masked_cmap('YlOrRd'), extent=extent,
                      vmin=0, vmax=np.nanpercentile(wd_show, 98),
                      zorder=3, interpolation='nearest')
    cb4 = plt.colorbar(im4, ax=ax4l, shrink=0.75, pad=0.03)
    cb4.set_label('归一化核密度', fontsize=10)
    cb4.outline.set_linewidth(0.6)
_clean_ax(ax4l, bounds)
add_north(ax4l)
add_scale(ax4l, bounds)
ax4l.set_title('(a) 历史积水点核密度空间分布', fontsize=11, fontweight='bold', pad=8)

# 右：散点位置 + 逐年数量柱状（若无点数据则改为核密度透明叠加）
ax4r = axes4[1]
if wl_density is not None:
    wd_show2 = ds_arr(wl_density, valid_mask)
    ax4r.imshow(wd_show2, cmap=masked_cmap('Blues'), extent=extent,
                vmin=0, vmax=np.nanpercentile(wd_show2, 98),
                zorder=2, interpolation='nearest', alpha=0.45)

if wl_pts is not None and len(wl_pts) > 0:
    # 按年份配色散点
    years_uniq = sorted(wl_pts['year'].unique())
    cmap_yr = plt.get_cmap('tab20', len(years_uniq))
    yr_color = {yr: cmap_yr(i) for i, yr in enumerate(years_uniq)}

    for yr in years_uniq:
        sub = wl_pts[wl_pts['year'] == yr]
        ax4r.scatter(sub['lon'], sub['lat'], s=6, color=yr_color[yr],
                     alpha=0.65, linewidths=0, zorder=5, label=str(yr))

    ax4r.legend(title='年份', fontsize=7, title_fontsize=8,
                loc='lower left', framealpha=0.9, ncol=2,
                markerscale=2.0, handletextpad=0.3)
    n_total_pts = len(wl_pts)
    ax4r.text(0.03, 0.97,
              f'积水点总数: {n_total_pts:,}\n年份数: {len(years_uniq)}',
              transform=ax4r.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.88, edgecolor='#AAAAAA'))
    ax4r.set_title('(b) 历史积水点位空间分布（按年份着色）', fontsize=11,
                   fontweight='bold', pad=8)
else:
    ax4r.set_title('(b) 历史积水点核密度（高分位数提取）', fontsize=11,
                   fontweight='bold', pad=8)
    # 提取高密度像元作为"热点"显示
    if wl_density is not None:
        vhigh = np.nanpercentile(wl_density[valid_mask & np.isfinite(wl_density)], 90)
        hot_rows, hot_cols = np.where((wl_density > vhigh) & valid_mask)
        if hot_rows.size > 0:
            hot_lons = transform.c + hot_cols * transform.a
            hot_lats = transform.f + hot_rows * transform.e
            ax4r.scatter(hot_lons[::50], hot_lats[::50], s=4,
                         color='#D7301F', alpha=0.4, linewidths=0, zorder=5)

_clean_ax(ax4r, bounds)
add_north(ax4r)
add_scale(ax4r, bounds)

fig4.savefig(os.path.join(OUT_DIR, 'Fig4_Waterlogging_Points.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig4_Waterlogging_Points.png")


# ============================================================
# Fig5  人口密度：多年均值空间分布 + 逐年时序
# ============================================================
print("[Fig5] 人口密度空间分布 + 逐年时序...")

pop_mean = load_tif(os.path.join(EXT_DIR, 'population_density_30m.tif'))

# 逐年人口均值
pop_ts_vals = []
for yr in STUDY_YEARS:
    fp = os.path.join(EXT_DIR, f'population_{yr}_30m.tif')
    arr = load_tif(fp)
    if arr is not None:
        v = arr[valid_mask & np.isfinite(arr)]
        pop_ts_vals.append(float(v.mean()) if v.size > 0 else np.nan)
    else:
        pop_ts_vals.append(np.nan)
pop_ts_vals = np.array(pop_ts_vals)

fig5 = plt.figure(figsize=(16, 7), facecolor='white')
gs5 = GridSpec(1, 2, figure=fig5, wspace=0.14,
               left=0.04, right=0.96, top=0.88, bottom=0.10)
fig5.suptitle('（二）社会经济数据预处理：人口密度\n'
              '（WorldPop 1km→30m双三次插值，2012—2024年均值）',
              fontsize=13, fontweight='bold')

ax5l = fig5.add_subplot(gs5[0, 0])
if city_gdf is not None:
    city_gdf.plot(ax=ax5l, color='#F5F5F5', zorder=1)
add_boundaries(ax5l, city_gdf, dist_gdf)
if pop_mean is not None:
    pop_show = ds_arr(pop_mean, valid_mask)
    im5 = ax5l.imshow(pop_show, cmap=masked_cmap('RdPu'), extent=extent,
                      vmin=np.nanpercentile(pop_show, 2),
                      vmax=np.nanpercentile(pop_show, 98),
                      zorder=3, interpolation='nearest')
    cb5 = plt.colorbar(im5, ax=ax5l, shrink=0.75, pad=0.03)
    cb5.set_label('归一化人口密度', fontsize=10)
    cb5.outline.set_linewidth(0.6)
_clean_ax(ax5l, bounds)
add_north(ax5l)
add_scale(ax5l, bounds)
ax5l.set_title('(a) 多年平均人口密度空间分布', fontsize=11, fontweight='bold', pad=8)

# 右：逐年折线
ax5r = fig5.add_subplot(gs5[0, 1])
valid_pop = ~np.isnan(pop_ts_vals)
if valid_pop.sum() > 0:
    ax5r.fill_between(years_arr[valid_pop], pop_ts_vals[valid_pop],
                      alpha=0.15, color='#7B2D8B')
    ax5r.plot(years_arr[valid_pop], pop_ts_vals[valid_pop],
              color='#7B2D8B', marker='o', linewidth=2.5, markersize=8,
              label='全域均值人口密度（归一化）')
    if valid_pop.sum() >= 3:
        z5 = np.polyfit(years_arr[valid_pop], pop_ts_vals[valid_pop], 1)
        p5 = np.poly1d(z5)
        ax5r.plot(years_arr, p5(years_arr), color='#D7301F', lw=1.8, linestyle='--',
                  alpha=0.8, label=f'趋势线 ({z5[0]:+.5f}/年)')
    ax5r.set_ylim(0, min(1.05, max(pop_ts_vals[valid_pop]) * 1.25))

ax5r.set_xlabel('年份', fontsize=11)
ax5r.set_ylabel('全域均值人口密度（归一化）', fontsize=11)
ax5r.set_xticks(years_arr)
ax5r.set_xticklabels([str(y) for y in STUDY_YEARS], rotation=35, fontsize=8.5)
ax5r.legend(fontsize=9, framealpha=0.9)
ax5r.grid(axis='y', alpha=0.3, linestyle='--')
ax5r.set_title('(b) 逐年人口密度全域均值时序', fontsize=11, fontweight='bold', pad=8)

fig5.savefig(os.path.join(OUT_DIR, 'Fig5_Population.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig5_Population.png")


# ============================================================
# Fig6  应急设施三图联排：避难场所 / 医院 / 消防站
# ============================================================
print("[Fig6] 应急设施核密度三图联排...")

poi_configs6 = [
    ('shelter_density_30m.tif',     'copper',  '应急避难场所密度',
     '(a) 应急避难场所核密度'),
    ('hospital_density_30m.tif',    'PuBu',    '综合医院密度',
     '(b) 综合医院核密度'),
    ('firestation_density_30m.tif', 'BuPu',    '消防救援站密度',
     '(c) 消防救援站核密度'),
]

fig6, axes6 = plt.subplots(1, 3, figsize=(21, 7), facecolor='white')
fig6.suptitle('（三）应急设施数据预处理：POI高斯核密度估计\n'
              '（GCJ-02→WGS84坐标转换，Silverman带宽，30m栅格）',
              fontsize=13, fontweight='bold')

for ax, (fname, cmap_name, cbar_label, title) in zip(axes6, poi_configs6):
    data = load_tif(os.path.join(EXT_DIR, fname))
    if city_gdf is not None:
        city_gdf.plot(ax=ax, color='#F5F5F5', zorder=1)
    add_boundaries(ax, city_gdf, dist_gdf)

    if data is not None:
        d_show = ds_arr(data, valid_mask)
        d_masked = np.ma.masked_where(
            ~np.isfinite(d_show) | (d_show < np.nanpercentile(d_show, 5)), d_show)
        im6 = ax.imshow(d_masked, cmap=masked_cmap(cmap_name), extent=extent,
                        vmin=0, vmax=np.nanpercentile(d_show, 98),
                        zorder=3, interpolation='nearest')
        cb6 = plt.colorbar(im6, ax=ax, shrink=0.75, pad=0.03,
                           orientation='horizontal')
        cb6.set_label(cbar_label, fontsize=9)
        cb6.outline.set_linewidth(0.6)
        cb6.ax.tick_params(labelsize=8)

        v6 = data[valid_mask & np.isfinite(data)]
        n_nonzero = int((v6 > 0.01).sum())
        ax.text(0.03, 0.97,
                f'覆盖像元: {n_nonzero:,}\n({n_nonzero/v6.size*100:.1f}%)',
                transform=ax.transAxes, fontsize=8.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor='#AAAAAA'))

    _clean_ax(ax, bounds)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)

add_north(axes6[2])
add_scale(axes6[2], bounds, bx0=0.55)

fig6.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, 'Fig6_Emergency_Facilities.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig6_Emergency_Facilities.png")


# ============================================================
# Fig7  数据对齐：有效像元掩码 + 10指标完整性热力图
# ============================================================
print("[Fig7] 数据对齐可视化...")

# ---- 左图：有效像元掩码空间图 ----
fig7 = plt.figure(figsize=(18, 7), facecolor='white')
gs7 = GridSpec(1, 2, figure=fig7, wspace=0.16,
               left=0.04, right=0.96, top=0.88, bottom=0.10)
fig7.suptitle('（四）数据对齐：全域有效像元掩码与10指标数据完整性\n'
              '（剔除任一指标缺值像元，北京市行政边界裁剪，30m栅格）',
              fontsize=13, fontweight='bold')

ax7l = fig7.add_subplot(gs7[0, 0])
if city_gdf is not None:
    city_gdf.plot(ax=ax7l, color='#EEEEEE', zorder=1)
add_boundaries(ax7l, city_gdf, dist_gdf)

valid_show = valid_mask[::DS, ::DS].astype(float)
# 区分：有效像元（绿）/ 无效像元（灰）/ 城市边界外（白）
cmap_mask = ListedColormap(['#FFFFFF', '#BDBDBD', '#2CA25F'])
cmap_mask.set_bad('white', alpha=0)

# 构建三值图：0=边界外，1=nodata，2=有效
nodata_show = nodata_mask[::DS, ::DS]
mask_vis = np.zeros_like(valid_show, dtype=np.float32)
mask_vis[:] = np.nan        # 边界外透明
# 找城市掩膜（用nodata_mask取反即近似）
mask_vis[~nodata_show] = 1  # 无效像元（边界内但nodata）
mask_vis[valid_show.astype(bool)] = 2  # 有效像元

ax7l.imshow(mask_vis, cmap=cmap_mask, extent=extent,
            vmin=0, vmax=2, zorder=3, interpolation='nearest')
patches7 = [
    mpatches.Patch(color='#2CA25F', label=f'有效像元 ({valid_mask.sum():,})'),
    mpatches.Patch(color='#BDBDBD', label=f'无效像元（含NoData）'),
]
ax7l.legend(handles=patches7, loc='lower left', fontsize=9.5, framealpha=0.92)

n_valid = int(valid_mask.sum())
total   = h * w
ax7l.text(0.97, 0.97,
          f'有效率: {n_valid/total*100:.1f}%\n'
          f'有效像元: {n_valid:,}\n'
          f'分辨率: 30m × 30m',
          transform=ax7l.transAxes, fontsize=9, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    alpha=0.88, edgecolor='#AAAAAA'))

_clean_ax(ax7l, bounds)
add_north(ax7l)
add_scale(ax7l, bounds)
ax7l.set_title('(a) 全域有效像元空间分布', fontsize=11, fontweight='bold', pad=8)

# ---- 右图：10指标完整性热力图 ----
ax7r = fig7.add_subplot(gs7[0, 1])

indicator_files = {
    '降雨量R':  (os.path.join(DYN_DIR,  'Precipitation_Mean.npy'), 'npy'),
    '径流系数CR': (os.path.join(DYN_DIR, 'CR_MultiYear_Mean.npy'), 'npy'),
    'TWI':       (os.path.join(STATIC_DIR,'twi.npy'),              'npy'),
    'HAND(m)':   (os.path.join(STATIC_DIR,'hand.npy'),             'npy'),
    '积水点WP':  (os.path.join(EXT_DIR,  'waterlogging_point_density_30m.tif'), 'tif'),
    '人口密度PD':(os.path.join(EXT_DIR,  'population_density_30m.tif'),         'tif'),
    '道路密度RD':(os.path.join(EXT_DIR,  'road_density_30m.tif'),               'tif'),
    '避难场所SH':(os.path.join(EXT_DIR,  'shelter_density_30m.tif'),            'tif'),
    '综合医院HO':(os.path.join(EXT_DIR,  'hospital_density_30m.tif'),           'tif'),
    '消防站FS':  (os.path.join(EXT_DIR,  'firestation_density_30m.tif'),        'tif'),
}

stat_rows = []
for ind_name, (fpath, ftype) in indicator_files.items():
    arr = load_npy(fpath) if ftype == 'npy' else load_tif(fpath)
    if arr is not None:
        v = arr[valid_mask & np.isfinite(arr)]
        valid_pct  = v.size / valid_mask.sum() * 100
        mean_val   = float(v.mean()) if v.size > 0 else np.nan
        std_val    = float(v.std())  if v.size > 0 else np.nan
        p5         = float(np.percentile(v, 5))  if v.size > 0 else np.nan
        p95        = float(np.percentile(v, 95)) if v.size > 0 else np.nan
    else:
        valid_pct = mean_val = std_val = p5 = p95 = np.nan

    stat_rows.append({
        '指标': ind_name,
        '完整性(%)': round(valid_pct, 2) if not np.isnan(valid_pct) else 0,
        '均值':   round(mean_val, 4) if not np.isnan(mean_val) else np.nan,
        '标准差': round(std_val, 4)  if not np.isnan(std_val)  else np.nan,
        'P5':     round(p5, 4)       if not np.isnan(p5)       else np.nan,
        'P95':    round(p95, 4)      if not np.isnan(p95)      else np.nan,
    })

df_stat = pd.DataFrame(stat_rows)

# 热力图：行=指标，列=统计维度（用归一化后的值显示颜色）
heat_cols = ['完整性(%)', '均值', '标准差', 'P5', 'P95']
heat_data = df_stat[heat_cols].copy()

# 按列做0-1归一化（仅用于颜色，数字保持原值）
heat_norm = heat_data.copy()
for col in heat_cols:
    col_vals = heat_norm[col].values.astype(float)
    lo, hi = np.nanmin(col_vals), np.nanmax(col_vals)
    if hi > lo:
        heat_norm[col] = (col_vals - lo) / (hi - lo)
    else:
        heat_norm[col] = 0.5

im7 = ax7r.imshow(heat_norm.values, cmap='YlGnBu', vmin=0, vmax=1,
                  aspect='auto', interpolation='nearest')

ax7r.set_xticks(range(len(heat_cols)))
ax7r.set_xticklabels(heat_cols, fontsize=10, fontweight='bold')
ax7r.set_yticks(range(len(df_stat)))
ax7r.set_yticklabels(df_stat['指标'].tolist(), fontsize=10)
ax7r.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

# 数值标注
for i in range(len(df_stat)):
    for j, col in enumerate(heat_cols):
        val = heat_data.iloc[i][col]
        if not np.isnan(val):
            txt_color = 'white' if heat_norm.iloc[i][col] > 0.65 else '#333333'
            fmt = '{:.1f}' if col == '完整性(%)' else '{:.4f}'
            ax7r.text(j, i, fmt.format(val), ha='center', va='center',
                      fontsize=8, color=txt_color, fontweight='bold')

ax7r.set_title('(b) 10项指标数据完整性与统计特征', fontsize=11, fontweight='bold', pad=22)

# 右侧colorbar
# cb7 = plt.colorbar(im7, ax=ax7r, shrink=0.8, pad=0.03)
# cb7.set_label('列内归一化值', fontsize=9)
# cb7.outline.set_linewidth(0.6)

# 添加圆形图例 (定义 5 个等级的代表颜色)
wl_colors = plt.get_cmap('RdPu')(np.linspace(0.2, 1.0, 5))
wl_labels = ['极低密度', '低密度', '中密度', '高密度', '极高密度']
add_circle_legend(ax4, wl_colors, wl_labels, "积水点核密度等级", loc='lower right')

# 加网格线
for i in range(len(df_stat) + 1):
    ax7r.axhline(i - 0.5, color='white', lw=1.0)
for j in range(len(heat_cols) + 1):
    ax7r.axvline(j - 0.5, color='white', lw=1.0)

fig7.savefig(os.path.join(OUT_DIR, 'Fig7_DataAlignment.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig7_DataAlignment.png")

# 顺手保存统计表
df_stat.to_csv(os.path.join(OUT_DIR, 'Table_Indicator_Completeness.csv'),
               index=False, encoding='utf-8-sig')
print("  ✅ Table_Indicator_Completeness.csv")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("V4.py 数据预处理可视化全部完成！")
print(f"  输出目录: {OUT_DIR}")
print("  Fig1_Precipitation.png       — 降雨量空间分布 + 逐年时序")
print("  Fig2_Runoff_LandCover.png    — 径流系数 + 三类下垫面")
print("  Fig3_Terrain_Hydrology.png   — DEM/坡度/TWI/HAND四图联排")
print("  Fig4_Waterlogging_Points.png — 历史积水点核密度")
print("  Fig5_Population.png          — 人口密度空间分布 + 逐年时序")
print("  Fig6_Emergency_Facilities.png— 三类应急设施核密度联排")
print("  Fig7_DataAlignment.png       — 有效像元掩码 + 完整性热力图")
print("  Table_Indicator_Completeness.csv")
print("=" * 70)
