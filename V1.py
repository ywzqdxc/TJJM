"""
指标空间分布图 v2.0
====================
修复：
  1. 数据严格裁剪到北京市轮廓内，轮廓外完全透明/白色
  2. 比例尺移至图框左下方（图外）
  3. 图例数值格式统一，去除混乱小数
  4. 整体风格更接近论文图5

依赖：
  pip install matplotlib numpy rasterio geopandas shapely
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import os, warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
STATIC_DIR = r'./Step_New/Static'
DYN_DIR    = r'./Step_New/Dynamic'
EXT_DIR    = r'./Step_New/External'
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
OUTPUT_DIR = r'./Step_New/Visualization/Indicator_Maps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ★ 北京市边界 Shapefile（必须设置，用于裁剪）
# 推荐：GADM China Level 0/1 提取北京，或阿里云DataV下载
BOUNDARY_SHP      = None   # 区县级
CITY_BOUNDARY_SHP = r'E:\Data\src\Beijing\北京市_市.shp'        # 全市轮廓

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# ============================================================
# 指标配置
# ============================================================
INDICATORS = [
    # 暴露度
    dict(label='(a) 汛期降雨量',        idx='a',
         file=os.path.join(DYN_DIR,    'Precipitation_Mean.npy'),
         is_tif=False, cmap='Blues',   n_class=5),
    dict(label='(b) 地形湿度指数 TWI',  idx='b',
         file=os.path.join(STATIC_DIR, 'twi.npy'),
         is_tif=False, cmap='GnBu',    n_class=5),
    dict(label='(c) 低洼度 HAND',       idx='c',
         file=os.path.join(STATIC_DIR, 'hand.npy'),
         is_tif=False, cmap='YlOrBr',  n_class=5),
    dict(label='(d) 历史积水点密度',    idx='d',
         file=os.path.join(EXT_DIR,    'waterlogging_point_density_30m.tif'),
         is_tif=True,  cmap='Purples', n_class=5),
    # 敏感性
    dict(label='(e) 人口密度',          idx='e',
         file=os.path.join(EXT_DIR,    'population_density_30m.tif'),
         is_tif=True,  cmap='RdPu',    n_class=5),
    dict(label='(f) 道路网络密度',      idx='f',
         file=os.path.join(EXT_DIR,    'road_density_30m.tif'),
         is_tif=True,  cmap='YlGn',    n_class=5),
    # 应对能力
    dict(label='(g) 应急避难场所密度',  idx='g',
         file=os.path.join(EXT_DIR,    'shelter_density_30m.tif'),
         is_tif=True,  cmap='copper',  n_class=5),
    dict(label='(h) 综合医院密度',      idx='h',
         file=os.path.join(EXT_DIR,    'hospital_density_30m.tif'),
         is_tif=True,  cmap='PuBu',    n_class=5),
    dict(label='(i) 消防救援站密度',    idx='i',
         file=os.path.join(EXT_DIR,    'firestation_density_30m.tif'),
         is_tif=True,  cmap='BuPu',    n_class=5),
]

# ============================================================
# Step1：加载基准网格 + 生成城市边界掩膜
# ============================================================
print("[1/3] 加载基准网格与边界...")

with rasterio.open(DEM_PATH) as ref:
    h, w          = ref.height, ref.width
    dst_crs       = ref.crs
    dst_transform = ref.transform
    dst_bounds    = ref.bounds

nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

# 加载 geopandas（必须）
try:
    import geopandas as gpd
    from shapely.geometry import mapping
    from shapely.ops import unary_union
    HAS_GPD = True
except ImportError:
    print("  ❌ 请安装 geopandas: pip install geopandas")
    HAS_GPD = False

# 生成城市轮廓掩膜（True=城市内部有效）
city_mask  = valid_mask.copy()   # fallback：用nodata_mask代替
boundary_gdf     = None
city_boundary_gdf = None
city_geoms        = None

if HAS_GPD:
    # 优先用全市轮廓
    shp_path = CITY_BOUNDARY_SHP if os.path.exists(CITY_BOUNDARY_SHP) else \
               (BOUNDARY_SHP     if os.path.exists(BOUNDARY_SHP)      else None)

    if shp_path:
        gdf_raw = gpd.read_file(shp_path).to_crs(dst_crs.to_epsg())

        # 合并为单一多边形（全市轮廓）
        city_union = unary_union(gdf_raw.geometry)
        city_geoms = [city_union]

        # 栅格化掩膜：使用 invert=True，直接让城市内部=True，外部=False
        from rasterio.features import geometry_mask as geo_mask

        city_mask = geo_mask(
            [mapping(city_union)],
            transform=dst_transform, out_shape=(h, w), invert=True
        )

        # 再与有效像元掩膜叠加
        city_mask = city_mask & valid_mask

        print(f"  ✅ 城市掩膜: {city_mask.sum():,} 个有效像元")

        # 区县级边界（用于白色分区线）
        if BOUNDARY_SHP and os.path.exists(BOUNDARY_SHP):
            boundary_gdf = gpd.read_file(BOUNDARY_SHP).to_crs(dst_crs.to_epsg())

        # 全市轮廓（用于黑色外轮廓）
        city_boundary_gdf = gdf_raw
    else:
        print(f"  ⚠️  未找到行政区划SHP，使用nodata_mask代替（数据不会被裁剪到精确轮廓）")
        print(f"      CITY_BOUNDARY_SHP = {CITY_BOUNDARY_SHP}")
        print(f"      BOUNDARY_SHP      = {BOUNDARY_SHP}")

# ============================================================
# Step2：加载所有指标数据
# ============================================================
print("[2/3] 加载指标数据...")

def load_arr(cfg, h, w, dst_crs, dst_transform, city_mask):
    """加载并裁剪到城市范围"""
    path = cfg['file']
    if not os.path.exists(path):
        print(f"  ❌ {cfg['label']}: 文件不存在 {path}")
        return None

    if not cfg['is_tif']:
        arr = np.load(path).astype(np.float32)
    else:
        with rasterio.open(path) as src:
            if src.height == h and src.width == w:
                arr = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
            else:
                arr = np.full((h, w), np.nan, dtype=np.float32)
                reproject(source=rasterio.band(src, 1), destination=arr,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs=dst_crs,
                          resampling=Resampling.bilinear, dst_nodata=np.nan)

    arr = np.where(arr < -1e10, np.nan, arr)
    # ★ 核心：城市外部置为 NaN
    arr = np.where(city_mask, arr, np.nan)

    v = arr[city_mask & np.isfinite(arr)]
    if v.size > 0:
        print(f"  ✅ {cfg['label']}: 有效={v.size:,}  "
              f"均值={v.mean():.4f}  范围=[{v.min():.4f},{v.max():.4f}]")
    else:
        print(f"  ⚠️  {cfg['label']}: 无有效数据")
    return arr


arrays = []
for cfg in INDICATORS:
    arrays.append(load_arr(cfg, h, w, dst_crs, dst_transform, city_mask))

# ============================================================
# Step3：绘图
# ============================================================
print("[3/3] 绘制图像...")

# 计算分位数断点（等频率分级，避免全0/全1问题）
def get_breaks(arr, city_mask, n=5):
    vals = arr[city_mask & np.isfinite(arr)]
    if vals.size < n:
        return np.linspace(0, 1, n + 1)
    q = np.linspace(0, 100, n + 1)
    bk = np.unique(np.percentile(vals, q))
    # 若unique后断点不够，补均匀断点
    if len(bk) < n + 1:
        bk = np.linspace(vals.min(), vals.max(), n + 1)
    bk[-1] *= 1.0001   # 稍微放大最大值，确保最大像元被包含
    return bk


def fmt(v):
    """智能格式化：自动选择合适精度"""
    if abs(v) >= 100:  return f'{v:.0f}'
    if abs(v) >= 10:   return f'{v:.1f}'
    if abs(v) >= 1:    return f'{v:.2f}'
    if abs(v) >= 0.01: return f'{v:.3f}'
    return f'{v:.4f}'


# ── 整体布局 ────────────────────────────────────────────────
# 3行×3列子图，每列左侧留图例空间，底部留比例尺空间
# 使用 GridSpec 精细控制间距

n_rows, n_cols = 3, 3
FIG_W, FIG_H   = 16, 13   # 英寸

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

# 主GridSpec：3行×3列，行高一致
gs = GridSpec(n_rows, n_cols, figure=fig,
              left=0.01, right=0.99, top=0.97, bottom=0.04,
              hspace=0.12, wspace=0.04)

# 计算图像显示范围
extent = [dst_bounds.left, dst_bounds.right,
          dst_bounds.bottom, dst_bounds.top]
ax_ratio = (dst_bounds.top - dst_bounds.bottom) / \
           (dst_bounds.right - dst_bounds.left)

for idx, (cfg, arr) in enumerate(zip(INDICATORS, arrays)):
    row, col = idx // n_cols, idx % n_cols
    ax = fig.add_subplot(gs[row, col])
    ax.set_aspect('equal')
    ax.set_facecolor('white')   # ★ 轴背景纯白

    if arr is None:
        ax.text(0.5, 0.5, '数据缺失\n请检查文件路径',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#999999')
        ax.set_title(cfg['label'], fontsize=10, pad=6)
        ax.axis('off')
        continue

    # 计算断点与离散色标
    n   = cfg['n_class']
    bks = get_breaks(arr, city_mask, n=n)
    base_cmap = plt.get_cmap(cfg['cmap'])
    colors    = [base_cmap(0.15 + 0.70 * i / (n - 1)) for i in range(n)]
    cmap_d    = mcolors.ListedColormap(colors)
    norm_d    = mcolors.BoundaryNorm(bks, ncolors=n)
    cmap_d.set_bad(color='white', alpha=0)   # ★ NaN完全透明

    # ── 绘制栅格 ──
    ds = 3   # 降采样（最终出图改为1）
    arr_ds = arr[::ds, ::ds]
    ax.imshow(arr_ds, cmap=cmap_d, norm=norm_d,
               extent=extent, origin='upper',
               interpolation='nearest', zorder=2)

    # ── 叠加区县白色分区线 ──
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(
            ax=ax, color='white', linewidth=0.5,
            alpha=0.8, zorder=4)

    # ── 叠加全市黑色外轮廓 ──
    if city_boundary_gdf is not None:
        city_boundary_gdf.boundary.plot(
            ax=ax, color='black', linewidth=1.0,
            alpha=1.0, zorder=5)
    elif city_geoms is None:
        # fallback：用nodata边界近似
        pass

    # ── 轴范围设为北京市边界范围 ──
    pad_x = (dst_bounds.right - dst_bounds.left) * 0.01
    pad_y = (dst_bounds.top - dst_bounds.bottom) * 0.01
    ax.set_xlim(dst_bounds.left  - pad_x, dst_bounds.right + pad_x)
    ax.set_ylim(dst_bounds.bottom - pad_y, dst_bounds.top  + pad_y)

    # ── 去除坐标轴 ──
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)   # ★ 无图框线，更干净

    # ── 图例（左上角，竖排色块）──
    LX, LY   = 0.02, 0.97   # 图例起点（axes fraction）
    BW, BH   = 0.04, 0.055  # 色块宽高
    GAP      = 0.004
    LABEL_FONT = 6.0

    # "指标归一化值"标题
    ax.text(LX, LY + 0.012, '指标归一化值',
            transform=ax.transAxes, fontsize=6.5, fontweight='bold',
            va='bottom', ha='left', color='#222222',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    for i, color in enumerate(colors):
        y_top = LY - i * (BH + GAP)
        y_bot = y_top - BH
        # 色块
        rect = plt.Rectangle(
            (LX, y_bot), BW, BH,
            facecolor=color, edgecolor='#AAAAAA', linewidth=0.4,
            transform=ax.transAxes, zorder=8
        )
        ax.add_patch(rect)
        # 数值标签（"lo~hi"格式）
        lo_s = fmt(bks[i])
        hi_s = fmt(bks[i + 1])
        label_txt = f'{lo_s}~{hi_s}'
        ax.text(LX + BW + 0.012, (y_top + y_bot) / 2,
                label_txt, transform=ax.transAxes,
                fontsize=LABEL_FONT, va='center', ha='left', color='#222222',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # ── 指北针（仅右上角子图，即 col=2, row=0）──
    if row == 0 and col == n_cols - 1:
        NX, NY = 0.88, 0.92
        ax.annotate('', xy=(NX, NY + 0.07), xytext=(NX, NY),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black',
                                   lw=1.5, mutation_scale=14))
        ax.text(NX, NY + 0.10, 'N', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── 比例尺（放在图框左下方，图外）──
    # 计算 30km 在经度方向对应的度数
    center_lat  = (dst_bounds.top + dst_bounds.bottom) / 2
    km_per_deg  = 111.32 * np.cos(np.radians(center_lat))
    bar_km      = 30
    bar_deg     = bar_km / km_per_deg
    total_deg_x = dst_bounds.right - dst_bounds.left
    bar_frac    = bar_deg / total_deg_x   # 比例尺在图宽的分数

    # 在子图正下方的figure坐标绘制比例尺
    # 仅在每列最后一行（row=2）显示
    if row == n_rows - 1:
        # 转换到figure坐标
        pos = ax.get_position()   # Bbox in figure fraction
        bar_x_start = pos.x0 + 0.01 * (pos.x1 - pos.x0)
        bar_x_end   = bar_x_start + bar_frac * (pos.x1 - pos.x0)
        bar_y       = pos.y0 - 0.022   # 在子图底部以下

        # 绘制比例尺线（使用fig.lines）
        line = plt.Line2D([bar_x_start, bar_x_end], [bar_y, bar_y],
                           transform=fig.transFigure,
                           color='black', linewidth=2,
                           solid_capstyle='butt', zorder=10)
        fig.add_artist(line)
        # 端点竖线
        for bx in [bar_x_start, bar_x_end]:
            tick = plt.Line2D([bx, bx],
                               [bar_y - 0.004, bar_y + 0.004],
                               transform=fig.transFigure,
                               color='black', linewidth=1.5, zorder=10)
            fig.add_artist(tick)
        # 标注文字
        fig.text((bar_x_start + bar_x_end) / 2, bar_y - 0.010,
                  f'0        {bar_km} km',
                  ha='center', va='top', fontsize=7,
                  transform=fig.transFigure)

    # ── 子图标题（底部居中）──
    ax.set_title(cfg['label'], fontsize=9.5, pad=5,
                  loc='center', fontweight='normal')

# ── 保存 ──────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'Indicator_Maps_v2.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight',
             facecolor='white', format='png',
             transparent=False)
plt.close()
print(f"\n✅ 保存完成: {out_path}")
print(f"   DPI=300  尺寸={FIG_W}×{FIG_H}英寸")
print(f"\n提示：当前 ds=3（降采样）。最终投稿前改为 ds=1 获得全分辨率。")