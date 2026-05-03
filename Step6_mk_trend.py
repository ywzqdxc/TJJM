"""
Step 6: Mann-Kendall趋势检验 + Sen's斜率估计
=============================================
输入：Vuln_TOPSIS_2012.npy ... Vuln_TOPSIS_2024.npy（Step3 v7.0输出）
输出：
  MK_Z_Score.tif       Z统计量空间栅格
  MK_P_Value.tif       p值空间栅格
  MK_Trend_Class.tif   趋势分类（1=显著上升 2=无趋势 3=显著下降）
  Sen_Slope.tif        Sen斜率（仅MK显著像元有值）
  MK_Trend_Map.png     趋势分类空间图
  Sen_Slope_Map.png    斜率空间图
  MK_Summary_Stats.csv 各类别面积统计

算法说明：
  Mann-Kendall检验（双侧，α=0.05）
    S  = Σ_{i<j} sgn(x[j]-x[i])
    Var(S) = T(T-1)(2T+5)/18
    Z  = (S-1)/√Var(S) if S>0
         (S+1)/√Var(S) if S<0
         0              if S=0
    p  = 2(1-Φ(|Z|))

  Sen's斜率
    β = median{(x[j]-x[i])/(j-i) : 0≤i<j≤T-1}

性能：向量化实现，2000万像元 × 78个时间对 ≈ 3-5分钟
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import norm as sci_norm
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
STATIC_DIR = r'./Step_New/Static'
RISK_DIR   = r'./Step_New/Risk_Map'
OUTPUT_DIR = r'./Step_New/Trend_Analysis'
VIS_DIR    = r'./Step_New/Visualization/Step6_MK'
for d in [OUTPUT_DIR, VIS_DIR]: os.makedirs(d, exist_ok=True)

STUDY_YEARS = list(range(2012, 2025))   # T=13年
ALPHA       = 0.05                       # 显著性水平

print("=" * 70)
print("Step 6: Mann-Kendall趋势检验 + Sen's斜率估计")
print(f"研究期: {STUDY_YEARS[0]}-{STUDY_YEARS[-1]}  T={len(STUDY_YEARS)}年")
print(f"显著性水平: α={ALPHA}")
print("=" * 70)

# ============================================================
# 一、加载基准信息与逐年数据
# ============================================================
print("\n[1/5] 加载基准网格与逐年脆弱性数据...")

with rasterio.open(DEM_PATH) as ref:
    h, w         = ref.height, ref.width
    out_profile  = ref.profile.copy()
out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='deflate')

nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

# 加载逐年脆弱性栅格
T = len(STUDY_YEARS)
year_arrays = []
missing_years = []

for year in STUDY_YEARS:
    fp = os.path.join(RISK_DIR, f'Vuln_TOPSIS_{year}.npy')
    if os.path.exists(fp):
        arr = np.load(fp).astype(np.float32)
        year_arrays.append(arr)
        v = arr[valid_mask & np.isfinite(arr)]
        print(f"  ✅ {year}: 有效={v.size:,}  均值={v.mean():.4f}")
    else:
        print(f"  ❌ {year}: Vuln_TOPSIS_{year}.npy 缺失，用NaN填充")
        year_arrays.append(np.full((h, w), np.nan, dtype=np.float32))
        missing_years.append(year)

if missing_years:
    print(f"\n  ⚠️  {len(missing_years)} 个年份缺失，这些像元将被标记为无效")
    print(f"     缺失年份: {missing_years}")
    print(f"     建议：先运行 step2_v7.py + step3_v7_annual_module.py")

# 构建时序矩阵 (T, H, W)
years_cube = np.stack(year_arrays, axis=0)   # shape: (13, H, W)
print(f"\n  时序矩阵: {years_cube.shape}  类型={years_cube.dtype}")

# 筛选完整像元（13年均有有效值）
complete_mask = valid_mask.copy()
for t in range(T):
    complete_mask &= np.isfinite(years_cube[t])

n_complete = int(complete_mask.sum())
n_valid    = int(valid_mask.sum())
print(f"  完整时序像元（13年均有效）: {n_complete:,} / {n_valid:,} "
      f"({n_complete/n_valid*100:.1f}%)")


# ============================================================
# 二、Mann-Kendall检验（向量化实现）
# ============================================================
print("\n[2/5] Mann-Kendall检验（向量化，处理所有完整像元）...")

# 提取完整像元的时序矩阵 (T, N)
# N = n_complete
flat_idx    = np.where(complete_mask.ravel())[0]   # 完整像元的展平索引
years_flat  = years_cube.reshape(T, -1)[:, flat_idx]   # (T, N)
N           = years_flat.shape[1]
print(f"  向量化计算中... (T={T}, N={N:,}  共 C({T},2)={T*(T-1)//2} 个时间对)")

# 计算 S = Σ_{i<j} sgn(x[j]-x[i])
S = np.zeros(N, dtype=np.int64)
for i in range(T - 1):
    for j in range(i + 1, T):
        diff = years_flat[j] - years_flat[i]
        S += np.sign(diff).astype(np.int64)

print(f"  S统计量计算完成  range=[{S.min()}, {S.max()}]")

# Var(S)（无连接情况）
var_S = T * (T - 1) * (2 * T + 5) / 18.0

# Z统计量
Z = np.where(S > 0, (S - 1) / np.sqrt(var_S),
    np.where(S < 0, (S + 1) / np.sqrt(var_S),
             np.zeros(N, dtype=np.float64)))

# 双侧p值
p_vals = 2.0 * (1.0 - sci_norm.cdf(np.abs(Z)))

# 趋势分类（1=显著上升, 2=无趋势, 3=显著下降）
trend_class = np.full(N, 2, dtype=np.int8)   # 默认无趋势
sig_up   = (p_vals < ALPHA) & (Z > 0)
sig_down = (p_vals < ALPHA) & (Z < 0)
trend_class[sig_up]   = 1
trend_class[sig_down] = 3

n_up   = int(sig_up.sum())
n_down = int(sig_down.sum())
n_none = N - n_up - n_down
print(f"\n  趋势检验结果（共 {N:,} 个完整像元）:")
print(f"    显著上升（p<{ALPHA}, Z>0）: {n_up:,}  ({n_up/N*100:.2f}%)")
print(f"    无显著趋势:                {n_none:,}  ({n_none/N*100:.2f}%)")
print(f"    显著下降（p<{ALPHA}, Z<0）: {n_down:,}  ({n_down/N*100:.2f}%)")

# ============================================================
# 三、Sen's斜率估计
# ============================================================
print("\n[3/5] Sen's斜率估计（分块计算，防内存溢出）...")

sen_slope = np.zeros(N, dtype=np.float32)

# 设置分块大小（每次处理 200 万个像元，内存占用约 600MB，非常安全）
chunk_size = 2_000_000
n_chunks = int(np.ceil(N / chunk_size))

for c in range(n_chunks):
    start_idx = c * chunk_size
    end_idx = min((c + 1) * chunk_size, N)

    # 提取当前块的时序矩阵 (T, chunk_size)
    years_chunk = years_flat[:, start_idx:end_idx]

    # 计算当前块的时间对斜率
    chunk_slopes = []
    for i in range(T - 1):
        for j in range(i + 1, T):
            dt = j - i
            chunk_slopes.append((years_chunk[j] - years_chunk[i]) / dt)

    # 堆叠当前块并求中位数
    slopes_mat_chunk = np.stack(chunk_slopes, axis=0)  # (78, chunk_size)
    sen_slope[start_idx:end_idx] = np.median(slopes_mat_chunk, axis=0).astype(np.float32)

    # 清理当前块的中间变量以释放内存
    del chunk_slopes, slopes_mat_chunk

    print(f"    进度: {c + 1}/{n_chunks} 块完成 ({(end_idx / N) * 100:.1f}%)")

# 仅对MK显著像元保留斜率，其余置NaN

# 仅对MK显著像元保留斜率，其余置NaN
sig_mask_flat = sig_up | sig_down
sen_slope_sig = np.where(sig_mask_flat, sen_slope, np.nan)

print(f"  Sen斜率统计（所有完整像元）:")
print(f"    均值={sen_slope.mean():.6f}/年  Std={sen_slope.std():.6f}")
print(f"    P5={np.percentile(sen_slope,5):.6f}  P95={np.percentile(sen_slope,95):.6f}")
print(f"  Sen斜率统计（仅MK显著像元）:")
sig_s = sen_slope_sig[np.isfinite(sen_slope_sig)]
if sig_s.size > 0:
    print(f"    均值={sig_s.mean():.6f}/年  Std={sig_s.std():.6f}")
    print(f"    最大上升={sig_s.max():.6f}  最大下降={sig_s.min():.6f}")


# ============================================================
# 四、还原到空间栅格并保存
# ============================================================
print("\n[4/5] 还原空间栅格并保存TIF...")

def flat_to_tif(flat_vals, flat_idx, h, w, path, profile, dtype=np.float32, nodata=-9999.0):
    grid = np.full(h * w, nodata, dtype=dtype)
    grid[flat_idx] = flat_vals.astype(dtype)
    grid_2d = grid.reshape(h, w)
    prof = profile.copy()
    if dtype == np.int8:
        prof.update(dtype=rasterio.int8, nodata=0)
    else:
        prof.update(dtype=rasterio.float32, nodata=nodata)
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(grid_2d, 1)
    return grid_2d

# Z统计量
Z_grid = flat_to_tif(Z.astype(np.float32), flat_idx, h, w,
                      os.path.join(OUTPUT_DIR, 'MK_Z_Score.tif'), out_profile)
print("  ✅ MK_Z_Score.tif")

# p值
p_grid = flat_to_tif(p_vals.astype(np.float32), flat_idx, h, w,
                      os.path.join(OUTPUT_DIR, 'MK_P_Value.tif'), out_profile)
print("  ✅ MK_P_Value.tif")

# 趋势分类
tc_grid = flat_to_tif(trend_class, flat_idx, h, w,
                       os.path.join(OUTPUT_DIR, 'MK_Trend_Class.tif'),
                       out_profile, dtype=np.int8)
print("  ✅ MK_Trend_Class.tif  (1=显著上升 2=无趋势 3=显著下降)")

# Sen斜率（显著像元）
sen_grid = flat_to_tif(sen_slope_sig, flat_idx, h, w,
                        os.path.join(OUTPUT_DIR, 'Sen_Slope.tif'), out_profile)
print("  ✅ Sen_Slope.tif  (仅MK显著像元，其余=-9999)")

# 汇总统计CSV
df_summary = pd.DataFrame([
    {'趋势类别': '显著上升（p<0.05）', '像元数': n_up,
     '占比%': n_up/N*100, '平均Sen斜率': float(sen_slope[sig_up].mean()) if n_up > 0 else np.nan},
    {'趋势类别': '无显著趋势',          '像元数': n_none,
     '占比%': n_none/N*100, '平均Sen斜率': float(sen_slope[~sig_up & ~sig_down].mean())},
    {'趋势类别': '显著下降（p<0.05）', '像元数': n_down,
     '占比%': n_down/N*100, '平均Sen斜率': float(sen_slope[sig_down].mean()) if n_down > 0 else np.nan},
    {'趋势类别': '合计（完整时序）',    '像元数': N,
     '占比%': 100.0, '平均Sen斜率': float(sen_slope.mean())},
])
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'MK_Summary_Stats.csv'),
                   index=False, encoding='utf-8-sig')
print("  ✅ MK_Summary_Stats.csv")


# ============================================================
# 五、可视化（学术出版级重构）
# ============================================================
print("\n[5/5] 生成学术出版级可视化图像...")

import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.colors as mcolors

# 1. 基础地理信息与比例尺参数
with rasterio.open(DEM_PATH) as ref:
    dst_bounds = ref.bounds
    center_lat = (dst_bounds.top + dst_bounds.bottom) / 2.0

extent = [dst_bounds.left, dst_bounds.right, dst_bounds.bottom, dst_bounds.top]
total_x = dst_bounds.right - dst_bounds.left
bar_km  = 30
bar_deg = bar_km / (111.32 * np.cos(np.radians(center_lat)))
bar_frac = bar_deg / total_x

# 2. 绘图辅助函数
def _clean_ax(ax, bounds, pad_ratio=0.02):
    pad_x = (bounds.right - bounds.left) * pad_ratio
    pad_y = (bounds.top - bounds.bottom) * pad_ratio
    ax.set_xlim(bounds.left - pad_x, bounds.right + pad_x)
    ax.set_ylim(bounds.bottom - pad_y, bounds.top + pad_y)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_facecolor('white')

def add_north_arrow(ax, x=0.92, y=0.88, size=0.08, fs=12):
    ax.annotate('', xy=(x, y+size), xytext=(x, y), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0, mutation_scale=15))
    ax.text(x, y+size+0.02, 'N', transform=ax.transAxes, ha='center', va='bottom',
            fontsize=fs, fontweight='bold')

def add_scale_bar(ax, bar_frac, bar_km, bx0=0.60, by=0.05, fs=10):
    bx1 = bx0 + bar_frac
    ax.plot([bx0, bx1], [by, by], transform=ax.transAxes, color='black', lw=3, zorder=10)
    for bx in [bx0, bx1]:
        ax.plot([bx, bx], [by-0.01, by+0.01], transform=ax.transAxes, color='black', lw=2, zorder=10)
    ax.text(bx0, by-0.025, '0', ha='center', va='top', transform=ax.transAxes, fontsize=fs)
    ax.text(bx1, by-0.025, f'{bar_km} km', ha='center', va='top', transform=ax.transAxes, fontsize=fs)

# 3. 加载行政边界
CITY_SHP = r'E:\Data\src\Beijing\北京市_市.shp'
DIST_SHP = r'E:\Data\src\Beijing\北京市_区.shp'

city_gdf = gpd.read_file(CITY_SHP) if os.path.exists(CITY_SHP) else None
dist_gdf = gpd.read_file(DIST_SHP) if os.path.exists(DIST_SHP) else None

def add_boundaries(ax):
    if dist_gdf is not None:
        dist_gdf.boundary.plot(ax=ax, color='white', linewidth=0.7, linestyle='--', alpha=0.8, zorder=4)
    if city_gdf is not None:
        city_gdf.boundary.plot(ax=ax, color='black', linewidth=1.2, zorder=5)

# 控制出图分辨率（1=最高精度原分辨率，2=平滑压缩降内存，建议出版用2）
DS = 2

# ── 图1：学术级 MK 趋势分类图 ─────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 9))
_clean_ax(ax1, dst_bounds)

# 绘制浅灰色的北京市底板
if city_gdf is not None:
    city_gdf.plot(ax=ax1, color='#F2F3F4', zorder=1)
add_boundaries(ax1)

# 【核心修复】：严格剔除无效数据 (确保仅保留 1, 2, 3)
tc_show = tc_grid[::DS, ::DS].astype(float)
tc_show[~np.isin(tc_show, [1, 2, 3])] = np.nan
tc_show_ma = np.ma.masked_invalid(tc_show) # 强制转为掩膜数组

# 定义学术配色：1=红(上升), 2=浅灰(无趋势), 3=蓝(下降)
cmap_tc = ListedColormap(['#E74C3C', '#F2F3F4', '#2C7BB6'])
norm_tc = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3)
cmap_tc.set_bad(color='none') # 【核心修复】：确保无效区完全透明

im1 = ax1.imshow(tc_show_ma, cmap=cmap_tc, norm=norm_tc, extent=extent, zorder=3, interpolation='nearest')

patches1 = [
    mpatches.Patch(color='#E74C3C', label='显著上升 (p<0.05)'),
    mpatches.Patch(color='#F2F3F4', label='无显著趋势'),
    mpatches.Patch(color='#2C7BB6', label='显著下降 (p<0.05)')
]
ax1.legend(handles=patches1, loc='upper left', fontsize=11, framealpha=0.95, edgecolor='#CCCCCC')

add_north_arrow(ax1)
add_scale_bar(ax1, bar_frac, bar_km)
ax1.set_title(f'北京市城市内涝脆弱性趋势分类图（Mann-Kendall检验, 2012-2024）\n'
              f'α={ALPHA}  双侧检验  T={T}年', fontsize=14, fontweight='bold', pad=15)

fig1.savefig(os.path.join(VIS_DIR, 'MK_Trend_Map.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ MK_Trend_Map.png (学术高定版)")


# ── 图2：学术级 Sen's 斜率图 ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 9))
_clean_ax(ax2, dst_bounds)

# 绘制浅灰色的北京市底板 (代表不显著区域)
if city_gdf is not None:
    city_gdf.plot(ax=ax2, color='#F2F3F4', zorder=1)
add_boundaries(ax2)

sen_show = sen_grid[::DS, ::DS].astype(float)
# 【核心修复】：将无数据区及未计算区转为 NaN
sen_show[sen_show <= -999.0] = np.nan
sen_show_ma = np.ma.masked_invalid(sen_show)

# 确定色标极值 (95%分位数)，保证正负渐变色对称
sig_s_plot = sen_show_ma.compressed() # 仅提取有效值计算极值
vmax_s = float(np.percentile(np.abs(sig_s_plot), 95)) if sig_s_plot.size > 10 else 0.01

im2 = ax2.imshow(sen_show_ma, cmap='RdBu_r', vmin=-vmax_s, vmax=vmax_s,
                 extent=extent, zorder=3, interpolation='nearest')

cbar = plt.colorbar(im2, ax=ax2, shrink=0.7, pad=0.03)
cbar.set_label("Sen's斜率（变化速率 / 年）", fontsize=12)
cbar.outline.set_linewidth(0.8)

add_north_arrow(ax2)
add_scale_bar(ax2, bar_frac, bar_km)
ax2.set_title("北京市城市内涝脆弱性 Sen's 斜率空间分布\n（仅显示MK显著像元，蓝=下降趋势，红=上升趋势）",
              fontsize=14, fontweight='bold', pad=15)

fig2.savefig(os.path.join(VIS_DIR, 'Sen_Slope_Map.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Sen_Slope_Map.png (学术高定版)")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("Step 6 (MK趋势检验) 完成！")
print("=" * 70)
print(f"  完整时序像元: {N:,}  T={T}年")
print(f"  显著上升: {n_up:,} ({n_up/N*100:.2f}%)")
print(f"  无显著趋势: {n_none:,} ({n_none/N*100:.2f}%)")
print(f"  显著下降: {n_down:,} ({n_down/N*100:.2f}%)")
print(f"\n  输出文件: {OUTPUT_DIR}")
print(f"    MK_Z_Score.tif / MK_P_Value.tif / MK_Trend_Class.tif")
print(f"    Sen_Slope.tif / MK_Summary_Stats.csv")
print(f"  可视化: {VIS_DIR}")
print(f"    MK_Trend_Map.png / Sen_Slope_Map.png / MK_Distribution.png")
print(f"\n  ⬇️  下一步：运行 step7_robustness.py（稳健性验证）")