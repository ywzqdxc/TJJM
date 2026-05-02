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
# 五、可视化
# ============================================================
print("\n[5/5] 生成可视化图像...")

DS = 4   # 降采样因子

def make_show(arr2d, nodata_val=-9999.0):
    """降采样 + 无效值转NaN，用于imshow"""
    d = arr2d[::DS, ::DS].astype(np.float32)
    d = np.where(d == nodata_val, np.nan, d)
    return d

# ── 图1：趋势分类图
fig1, ax1 = plt.subplots(figsize=(12, 8))
tc_show = make_show(tc_grid, nodata_val=0)

trend_colors = ['#BDBDBD',   # 0=无效/背景
                '#D7191C',   # 1=显著上升（红）
                '#F0F0F0',   # 2=无趋势（浅灰）
                '#2C7BB6']   # 3=显著下降（蓝）
cmap_tc = ListedColormap(trend_colors)
cmap_tc.set_bad('white', 0.0)

im1 = ax1.imshow(tc_show, cmap=cmap_tc, vmin=0, vmax=3,
                  interpolation='nearest')
ax1.axis('off')
patches1 = [
    mpatches.Patch(color='#D7191C', label='显著上升'),
    mpatches.Patch(color='#F0F0F0', label='无显著趋势'),
    mpatches.Patch(color='#2C7BB6', label='显著下降'),
]
ax1.legend(handles=patches1, loc='upper left', fontsize=11, framealpha=0.9)
ax1.set_title(f'北京市城市内涝脆弱性趋势分类图（Mann-Kendall检验, 2012-2024）\n'
               f'α={ALPHA}  双侧检验  T={T}年  完整像元{N:,}个',
               fontsize=13, fontweight='bold', pad=15)
fig1.savefig(os.path.join(VIS_DIR, 'MK_Trend_Map.png'),
              dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ MK_Trend_Map.png")

# ── 图2：Sen斜率图（发散色标，仅显著像元）
fig2, ax2 = plt.subplots(figsize=(12, 8))

# =========================================================
# 【新增】：加载并生成严格的北京市行政边界掩膜
# =========================================================
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.ops import unary_union
from shapely.geometry import mapping

CITY_BOUNDARY_SHP = r'E:\Data\src\Beijing\北京市_市.shp'
try:
    # 加载 SHP 边界，并转换为与栅格相同的坐标系
    city_gdf = gpd.read_file(CITY_BOUNDARY_SHP).to_crs(out_profile['crs'])
    city_union = unary_union(city_gdf.geometry)
    # 生成精确的布尔掩膜
    city_vis = geometry_mask([mapping(city_union)], transform=out_profile['transform'],
                             out_shape=(h, w), invert=True)
    # 结合现有的 valid_mask
    bg_mask = city_vis & valid_mask
except Exception as e:
    print(f"  ⚠️ 边界加载失败，使用默认矩形掩膜: {e}")
    bg_mask = valid_mask

# 1. 绘制精确的北京市浅灰色底图
bg_grid = np.where(bg_mask, 1.0, np.nan)
bg_show = make_show(bg_grid, nodata_val=np.nan)
ax2.imshow(bg_show, cmap=ListedColormap(['#E5E5E5']), interpolation='nearest')

# 2. 加载并处理需要叠加的斜率数据
sen_show = make_show(sen_grid, nodata_val=-9999.0)

# 确定色标范围（基于95%分位数）
sig_s_plot = sen_show[np.isfinite(sen_show)]
if sig_s_plot.size > 10:
    vmax_s = float(np.percentile(np.abs(sig_s_plot), 95))
else:
    vmax_s = 0.01

# 3. 将彩色的斜率点精准叠加到底图上
im2 = ax2.imshow(sen_show, cmap='RdBu_r', vmin=-vmax_s, vmax=vmax_s,
                  interpolation='bilinear')
ax2.axis('off')
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
cbar2.set_label("Sen's斜率（脆弱性/年）", fontsize=12)
ax2.set_title(f"北京市城市内涝脆弱性 Sen's 斜率空间分布\n"
               f"（仅显示MK显著像元，蓝=下降趋势，红=上升趋势）",
               fontsize=13, fontweight='bold', pad=15)
fig2.savefig(os.path.join(VIS_DIR, 'Sen_Slope_Map.png'),
              dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Sen_Slope_Map.png")

# ── 图3：Z分布直方图 + 显著性阈值线
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
z_thresh = sci_norm.ppf(1 - ALPHA / 2)

axes3[0].hist(Z, bins=100, color='#4575b4', alpha=0.7, edgecolor='white')
axes3[0].axvline( z_thresh, color='#d73027', linewidth=2, linestyle='--',
                   label=f'+{z_thresh:.2f} (α={ALPHA}双侧)')
axes3[0].axvline(-z_thresh, color='#d73027', linewidth=2, linestyle='--',
                   label=f'-{z_thresh:.2f}')
axes3[0].set_xlabel('Z统计量', fontsize=12)
axes3[0].set_ylabel('像元数', fontsize=12)
axes3[0].set_title('MK检验 Z统计量分布', fontsize=12, fontweight='bold')
axes3[0].legend(fontsize=10)
axes3[0].grid(alpha=0.3)

axes3[1].hist(sen_slope * 1000, bins=100, color='#d73027', alpha=0.7, edgecolor='white')
axes3[1].axvline(0, color='black', linewidth=1.5)
axes3[1].set_xlabel("Sen's斜率（×10⁻³/年）", fontsize=12)
axes3[1].set_ylabel('像元数', fontsize=12)
axes3[1].set_title("Sen's斜率分布（所有完整像元）", fontsize=12, fontweight='bold')
axes3[1].grid(alpha=0.3)

plt.suptitle("MK趋势检验统计量分布（北京市 2012-2024）", fontsize=14, fontweight='bold')
plt.tight_layout()
fig3.savefig(os.path.join(VIS_DIR, 'MK_Distribution.png'),
              dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ MK_Distribution.png")


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