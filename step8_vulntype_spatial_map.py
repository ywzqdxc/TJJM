"""
step8_vulntype_spatial_map.py
==============================
读取 E/S/C 三维得分栅格，按中位数阈值划分 8 种致脆类型，
绘制精美空间分布图并输出 PNG。

输入：
  ./Step_New/Static/nodata_mask.npy
  ./Step_New/Risk_Map/Exposure_Score.npy
  ./Step_New/Risk_Map/Sensitivity_Score.npy
  ./Step_New/Risk_Map/CopingCapacity_Score.npy
  E:/Data/src/DEM数据/北京市_DEM_30m分辨率_NASA数据.tif  （仅用于获取空间参考）
  E:/Data/src/Beijing/北京市_市.shp
  E:/Data/src/Beijing/北京市_区.shp

输出：
  ./Step_New/Visualization/Step8_VulnType_Spatial_Map.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
STATIC_DIR = r'./Step_New/Static'
RISK_DIR   = r'./Step_New/Risk_Map'
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
CITY_SHP   = r'E:\Data\src\Beijing\北京市_市.shp'
DIST_SHP   = r'E:\Data\src\Beijing\北京市_区.shp'
OUT_DIR    = r'./Step_New/Visualization'
os.makedirs(OUT_DIR, exist_ok=True)

# 降采样率（与其他脚本保持一致，可按需调整）
DS = 4

# ============================================================
# 8 种致脆类型定义
# ============================================================
# 编码规则：type_id = E_flag*4 + S_flag*2 + C_flag
# E=0,S=0,C=0 → 0 (O)
# E=1,S=0,C=0 → 4  ← 但为了与说明书对齐，手动映射
TYPE_LABELS = ['O',  'E',   'S',   'C',   'ES',  'EC',  'SC',  'ESC']
TYPE_NAMES  = [
    'O（弱综合型）',
    'E（暴露致险）',
    'S（敏感致险）',
    'C（应对不足）',
    'ES（暴露-敏感）',
    'EC（暴露-应对）',
    'SC（敏感-应对）',
    'ESC（强综合型）',
]
TYPE_COLORS = [
    '#5DB141',  # O  — 绿色
    '#CB2926',  # E  — 红色
    '#DF5943',  # S  — 橘红色
    '#779BBF',  # C  — 灰蓝色
    '#71469D',  # ES — 紫色
    '#DEC3A2',  # EC — 米棕色
    '#E7BB69',  # SC — 土黄色
    '#7EB178',  # ESC— 灰绿色
]

# E/S/C 高低组合 → type index 0..7（按说明书顺序）
# (E_flag, S_flag, C_flag) → type_id
_ESC_TO_ID = {
    (0, 0, 0): 0,  # O
    (1, 0, 0): 1,  # E
    (0, 1, 0): 2,  # S
    (0, 0, 1): 3,  # C
    (1, 1, 0): 4,  # ES
    (1, 0, 1): 5,  # EC
    (0, 1, 1): 6,  # SC
    (1, 1, 1): 7,  # ESC
}

# ============================================================
# 工具函数
# ============================================================
def load_npy(path):
    if not os.path.exists(path):
        print(f'  [警告] 文件不存在: {path}')
        return None
    return np.load(path).astype(np.float32)


def clip_imshow(im_obj, ax, city_gdf):
    """用城市边界多边形裁剪 imshow，避免颜色溢出到矩形 extent 之外。"""
    if city_gdf is None:
        return
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    import shapely.ops
    merged = shapely.ops.unary_union(city_gdf.geometry)

    def _geom_to_path(geom):
        polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        verts, codes = [], []
        for poly in polys:
            ext = np.array(poly.exterior.coords)
            verts.append(ext)
            codes += [Path.MOVETO] + [Path.LINETO] * (len(ext) - 2) + [Path.CLOSEPOLY]
            for hole in poly.interiors:
                h = np.array(hole.coords)
                verts.append(h)
                codes += [Path.MOVETO] + [Path.LINETO] * (len(h) - 2) + [Path.CLOSEPOLY]
        return Path(np.vstack(verts), codes)

    path = _geom_to_path(merged)
    patch = PathPatch(path, transform=ax.transData)
    im_obj.set_clip_path(patch)


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
    ax.plot([bx0, bx1], [by, by], transform=ax.transAxes,
            color='black', lw=3, zorder=10)
    for bx in [bx0, bx1]:
        ax.plot([bx, bx], [by - 0.012, by + 0.012], transform=ax.transAxes,
                color='black', lw=2, zorder=10)
    ax.text(bx0, by - 0.03, '0', ha='center', va='top',
            transform=ax.transAxes, fontsize=fs)
    ax.text(bx1, by - 0.03, f'{bar_km} km', ha='center', va='top',
            transform=ax.transAxes, fontsize=fs)


# ============================================================
# 1. 加载空间参考（DEM bounds / transform）
# ============================================================
print('=' * 60)
print('step8_vulntype_spatial_map.py')
print('=' * 60)

print('\n[1] 读取空间参考...')
with rasterio.open(DEM_PATH) as ref:
    h, w      = ref.height, ref.width
    bounds    = ref.bounds
    transform = ref.transform
    crs       = ref.crs
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
print(f'    栅格尺寸: {h} x {w},  CRS: {crs}')

# ============================================================
# 2. 加载掩膜与三维得分
# ============================================================
print('\n[2] 读取掩膜和 E/S/C 得分...')
nodata_mask = load_npy(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

E_arr = load_npy(os.path.join(RISK_DIR, 'Exposure_Score.npy'))
S_arr = load_npy(os.path.join(RISK_DIR, 'Sensitivity_Score.npy'))
C_arr = load_npy(os.path.join(RISK_DIR, 'CopingCapacity_Score.npy'))

for name, arr in [('Exposure_Score', E_arr), ('Sensitivity_Score', S_arr),
                  ('CopingCapacity_Score', C_arr)]:
    if arr is None:
        raise FileNotFoundError(f'缺少必须文件: {name}.npy')

# ============================================================
# 3. 计算中位数阈值，生成 E/S/C 0/1 标志
# ============================================================
print('\n[3] 计算中位数阈值...')
vm = valid_mask & np.isfinite(E_arr) & np.isfinite(S_arr) & np.isfinite(C_arr)

med_E = float(np.median(E_arr[vm]))
med_S = float(np.median(S_arr[vm]))
med_C = float(np.median(C_arr[vm]))
print(f'    中位数 — E: {med_E:.4f}  S: {med_S:.4f}  C: {med_C:.4f}')

E_flag = (E_arr > med_E).astype(np.uint8)
S_flag = (S_arr > med_S).astype(np.uint8)
C_flag = (C_arr > med_C).astype(np.uint8)

# ============================================================
# 4. 逐像元赋予类型编号 0~7
# ============================================================
print('\n[4] 计算像元类型...')
type_map = np.full((h, w), -1, dtype=np.int8)
for (ef, sf, cf), tid in _ESC_TO_ID.items():
    sel = vm & (E_flag == ef) & (S_flag == sf) & (C_flag == cf)
    type_map[sel] = tid

# 类型统计
print('    各类型像元数量:')
for tid, lbl in enumerate(TYPE_LABELS):
    cnt = int((type_map == tid).sum())
    pct = cnt / vm.sum() * 100
    print(f'      {lbl:4s}: {cnt:>8,}  ({pct:.1f}%)')

# ============================================================
# 5. 降采样（保持与其他图一致的出图速度）
# ============================================================
type_ds = type_map[::DS, ::DS].astype(np.float32)
type_ds[type_ds < 0] = np.nan            # 无效像元 → NaN → 透明

# ============================================================
# 6. 加载边界矢量
# ============================================================
print('\n[5] 加载行政边界...')
try:
    import geopandas as gpd
    city_gdf = gpd.read_file(CITY_SHP) if os.path.exists(CITY_SHP) else None
    dist_gdf = gpd.read_file(DIST_SHP) if os.path.exists(DIST_SHP) else None
    print('    ✅ 行政边界加载成功')
except Exception as e:
    city_gdf = dist_gdf = None
    print(f'    ⚠️  边界加载失败: {e}')

# ============================================================
# 7. 绘图
# ============================================================
print('\n[6] 绘图...')

fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
fig.suptitle('北京市城市洪涝脆弱性致脆类型空间分布\n'
             '（基于E/S/C中位数阈值划分，8种致脆类型）',
             fontsize=14, fontweight='bold', y=0.97)

# 底色：北京范围填灰，区界范围外白
if city_gdf is not None:
    city_gdf.plot(ax=ax, color='#F0F0F0', zorder=1)

# 区县边界
if dist_gdf is not None:
    dist_gdf.boundary.plot(ax=ax, color='#555555', linewidth=0.5, zorder=5)
# 城市外轮廓
if city_gdf is not None:
    city_gdf.boundary.plot(ax=ax, color='#333333', linewidth=1.0, zorder=6)

# 类型色块
cmap_type = ListedColormap(TYPE_COLORS)
norm_type  = BoundaryNorm(np.arange(-0.5, 8, 1), ncolors=8)
cmap_type.set_bad('white', alpha=0)

im = ax.imshow(type_ds, cmap=cmap_type, norm=norm_type,
               extent=extent, zorder=3, interpolation='nearest')
clip_imshow(im, ax, city_gdf)

# ---- 坐标轴清理 ----
px = (bounds.right - bounds.left) * 0.01
py = (bounds.top - bounds.bottom) * 0.01
ax.set_xlim(bounds.left - px, bounds.right + px)
ax.set_ylim(bounds.bottom - py, bounds.top + py)
ax.axis('off')
ax.set_facecolor('white')

# ---- 图例（两列：左列 O/E/S/C，右列 ES/EC/SC/ESC）----
left_handles  = [mpatches.Patch(color=TYPE_COLORS[i], label=TYPE_NAMES[i])
                 for i in range(4)]
right_handles = [mpatches.Patch(color=TYPE_COLORS[i], label=TYPE_NAMES[i])
                 for i in range(4, 8)]
all_handles   = left_handles + right_handles

legend = ax.legend(
    handles=all_handles,
    title='内涝风险类型',
    title_fontsize=10,
    fontsize=9,
    loc='upper left',
    ncol=2,
    frameon=False,
    handlelength=1.4,
    handleheight=1.4,
    handletextpad=0.6,
    columnspacing=1.2,
    labelspacing=0.55,
    borderpad=0.5,
)
legend.get_title().set_fontweight('bold')

# ---- 指北针 ----
add_north(ax, x=0.92, y=0.85)

# ---- 比例尺 ----
add_scale(ax, bounds, bar_km=30, bx0=0.60, by=0.05)

# # ---- 像元统计文本框 ----
# total_valid = int(vm.sum())
# stat_lines = [f'有效像元总数: {total_valid:,}']
# for tid, lbl in enumerate(TYPE_LABELS):
#     cnt = int((type_map == tid).sum())
#     pct = cnt / total_valid * 100
#     stat_lines.append(f'{lbl}: {cnt:,} ({pct:.1f}%)')
# ax.text(0.98, 0.03, '\n'.join(stat_lines),
#         transform=ax.transAxes, fontsize=7.5, va='bottom', ha='right',
#         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
#                   alpha=0.88, edgecolor='#AAAAAA'))

# ============================================================
# 8. 保存
# ============================================================
out_path = os.path.join(OUT_DIR, 'Step8_VulnType_Spatial_Map.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f'\n✅ 已保存: {out_path}')
print('=' * 60)
