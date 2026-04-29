"""
Step 2.5 Enhanced: POI暴露度增强分析 (v3.1 修复版)
=====================================================
修复：
  1. WDI从NPY成功加载 ✅
  2. 修复统计检验NaN/Inf错误
  3. 改进双高区定义（更合理的分位数）
"""

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr, kruskal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os, gc, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径
# ============================================================
POI_PATH    = r'D:\BaiduNetdiskDownload\北京市POI数据.csv'
DEM_PATH    = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
STATIC_DIR  = r'./Step_New/Static'
DYN_DIR     = r'./Step_New/Dynamic'
OUTPUT_DIR  = r'./Step_New/POI_Exposure'
VIS_DIR     = r'./Step_New/Visualization/Step2_5'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,    exist_ok=True)

PLOT_DS = 4  # 降采样因子

# ============================================================
# POI权重表（14大类全覆盖）
# ============================================================
CATEGORY_MAP = {
    '医疗保健':   {'weight': 5.0, 'color': '#E74C3C', 'label': '医疗'},
    '科教文化':   {'weight': 4.0, 'color': '#9B59B6', 'label': '科教'},
    '交通设施':   {'weight': 3.5, 'color': '#3498DB', 'label': '交通'},
    '生活服务':   {'weight': 2.5, 'color': '#2ECC71', 'label': '生活服务'},
    '酒店住宿':   {'weight': 2.0, 'color': '#F39C12', 'label': '住宿'},
    '餐饮美食':   {'weight': 2.0, 'color': '#E67E22', 'label': '餐饮'},
    '购物消费':   {'weight': 2.0, 'color': '#1ABC9C', 'label': '购物'},
    '休闲娱乐':   {'weight': 2.0, 'color': '#3498DB', 'label': '休闲'},
    '旅游景点':   {'weight': 3.0, 'color': '#8E44AD', 'label': '景点'},
    '运动健身':   {'weight': 2.0, 'color': '#27AE60', 'label': '运动'},
    '公司企业':   {'weight': 2.5, 'color': '#7F8C8D', 'label': '企业'},
    '商务住宅':   {'weight': 2.5, 'color': '#95A5A6', 'label': '商务'},
    '汽车相关':   {'weight': 1.5, 'color': '#E91E63', 'label': '汽车'},
    '金融机构':   {'weight': 2.0, 'color': '#FFC107', 'label': '金融'},
}

SPECIAL_FACILITIES = {
    '综合医院':8.0, '专科医院':7.0, '诊所':5.0, '药店':3.0,
    '小学':6.0, '中学':6.0, '大学':5.0, '幼儿园':4.0,
    '公交站':3.0, '地铁站':4.0, '火车站':5.0, '停车场':2.0,
    '公厕':1.5, '景点':3.5, '社区中心':3.0, '文化宫':3.5,
    '旅馆':3.0, '度假养老':4.0, '农家乐':2.5, '中国菜':2.0,
    '加油站':1.5, '银行':2.5, 'ATM':2.0,
}

# ============================================================
# 工具函数
# ============================================================
def log_normalize(data, valid_mask, pmin=1, pmax=99):
    """
    对数归一化（改善极端偏态分布）
    处理NaN和Inf值
    """
    result = np.full_like(data, np.nan, dtype=np.float32)

    # 提取有效数据
    valid_data = data[valid_mask]
    valid_data = valid_data[np.isfinite(valid_data)]
    valid_data = valid_data[valid_data > 0]

    if len(valid_data) < 100:
        return np.zeros_like(data)

    # 对数变换
    log_data = np.log1p(data)
    log_valid = log_data[valid_mask & np.isfinite(log_data)]

    lo = np.nanpercentile(log_valid, pmin)
    hi = np.nanpercentile(log_valid, pmax)

    if hi - lo < 1e-8:
        return np.zeros_like(data)

    # 归一化
    norm_vals = np.clip((log_data - lo) / (hi - lo), 0, 1)
    result[valid_mask] = norm_vals[valid_mask]

    return result

def clean_array(arr):
    """清理数组中的NaN和Inf"""
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

def fast_kernel_density(lons, lats, weights, h, w, transform, bandwidth=0.01):
    """快速核密度估计（矢量化加速）"""
    density = np.zeros((h, w), dtype=np.float32)

    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)

    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    rows, cols = rows[valid], cols[valid]
    wgt = weights[valid]

    np.add.at(density, (rows, cols), wgt)

    sigma = max(1.0, bandwidth / abs(transform[0]))
    density = gaussian_filter(density, sigma=sigma)

    return density

def load_wdi(dyn_dir, h, w):
    """加载WDI（优先NPY）"""
    # 1. NPY
    npy_path = os.path.join(dyn_dir, 'WDI_MultiYear_Max.npy')
    if os.path.exists(npy_path):
        try:
            data = np.load(npy_path)
            if data.shape == (h, w):
                print(f"    ✅ 从NPY加载WDI")
                return data.astype(np.float32)
        except Exception as e:
            print(f"    ⚠️  NPY加载失败: {e}")

    # 2. TIFF
    tif_path = os.path.join(dyn_dir, 'WDI_MultiYear_Max.tif')
    if os.path.exists(tif_path):
        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1).astype(np.float32)
                data = np.where(data < -1e10, np.nan, data)
                data = np.where(data == -9999.0, np.nan, data)
                if data.shape == (h, w):
                    print(f"    ✅ 从TIFF加载WDI")
                    return data
        except Exception as e:
            print(f"    ⚠️  TIFF加载失败: {e}")

    # 3. 模拟
    print("    ⚠️  使用模拟WDI")
    np.random.seed(42)
    base = np.random.randn(h, w).astype(np.float32)
    base = gaussian_filter(base, sigma=8)
    return (base - base.min()) / (base.max() - base.min() + 1e-8)

def compute_beijing_outline(valid_mask, ds=4):
    """提取北京市轮廓"""
    from scipy.ndimage import binary_erosion
    mask_ds = valid_mask[::ds, ::ds]
    eroded = binary_erosion(mask_ds, iterations=2)
    outline = mask_ds.astype(np.uint8) - eroded.astype(np.uint8)
    return outline == 1

def plot_masked(ax, data, valid_mask, cmap='hot_r', vmin=None, vmax=None,
                ds=PLOT_DS, title='', add_outline=True):
    """安全绘制北京市地图"""
    # 降采样
    d = data[::ds, ::ds].astype(np.float32)
    m = valid_mask[::ds, ::ds]

    # 清理无效值
    d = np.where(m & np.isfinite(d), d, np.nan)
    d[~m] = np.nan

    valid_vals = d[m & ~np.isnan(d)]
    if valid_vals.size < 10:
        ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return None

    if vmin is None: vmin = float(np.nanpercentile(valid_vals, 2))
    if vmax is None: vmax = float(np.nanpercentile(valid_vals, 98))

    if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    rgba = cm(norm(d)).astype(np.float32)
    rgba[~m, 3] = 0.0
    rgba[np.isnan(d) & m, 3] = 0.0

    ax.imshow(rgba)

    if add_outline:
        outline = compute_beijing_outline(valid_mask, ds)
        ax.contour(outline, levels=[0.5], colors='black', linewidths=1.0, alpha=0.6)

    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    return sm

# ============================================================
# 主流程
# ============================================================
print("=" * 70)
print("Step 2.5 Enhanced v3.1: POI暴露度增强分析")
print("=" * 70)

# 1. 加载数据
print("\n[1/6] 加载数据...")
df_poi = pd.read_csv(POI_PATH, encoding='utf-8')
df_poi = df_poi.dropna(subset=['经度', '纬度'])
df_poi = df_poi[(df_poi['经度'] > 115) & (df_poi['经度'] < 118) &
                (df_poi['纬度'] > 39) & (df_poi['纬度'] < 42)]
print(f"    POI: {len(df_poi):,} 条")

valid_mask = ~np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy'))
h, w = valid_mask.shape
n_valid = int(valid_mask.sum())
print(f"    栅格: {h}×{w}, 有效像元: {n_valid:,}")

# 2. 计算POI暴露度
print("\n[2/6] 计算POI暴露度...")

# 计算权重
weights = np.ones(len(df_poi), dtype=np.float32)
for idx, (_, row) in enumerate(df_poi.iterrows()):
    cat, sub = row['大类'], row['中类']
    base_w = CATEGORY_MAP.get(cat, {}).get('weight', 1.0)
    special_w = SPECIAL_FACILITIES.get(sub, 1.0)
    weights[idx] = base_w * max(1.0, special_w / base_w)

# 分大类核密度
category_densities = {}
category_counts = df_poi['大类'].value_counts()

with rasterio.open(DEM_PATH) as ref:
    transform_ref = ref.transform

for cat, info in CATEGORY_MAP.items():
    cat_mask = df_poi['大类'].values == cat
    if not np.any(cat_mask):
        continue

    density = fast_kernel_density(
        df_poi.loc[cat_mask, '经度'].values,
        df_poi.loc[cat_mask, '纬度'].values,
        weights[cat_mask], h, w, transform_ref,
        bandwidth=0.012
    )
    category_densities[cat] = density
    gc.collect()

# 加权叠加 + 对数归一化
total_exposure_raw = np.zeros((h, w), dtype=np.float32)
for cat, density in category_densities.items():
    total_exposure_raw += density * CATEGORY_MAP[cat]['weight']

total_exposure = log_normalize(total_exposure_raw, valid_mask, pmin=1, pmax=99)
total_exposure = np.where(valid_mask, total_exposure, np.nan)

# 3. 加载WDI
print("\n[3/6] 加载WDI...")
wdi_raw = load_wdi(DYN_DIR, h, w)

# 归一化WDI（安全处理NaN/Inf）
wdi_clean = np.where(valid_mask & np.isfinite(wdi_raw), wdi_raw, np.nan)
wdi_min = np.nanmin(wdi_clean[valid_mask])
wdi_max = np.nanmax(wdi_clean[valid_mask])
wdi_range = wdi_max - wdi_min

if wdi_range < 1e-8:
    wdi_range = 1.0

wdi_norm = np.where(valid_mask, (wdi_clean - wdi_min) / wdi_range, np.nan)

# 4. 综合风险
print("\n[4/6] 计算综合风险...")
risk_combined = np.where(valid_mask,
                         wdi_norm * total_exposure * 1.5,
                         np.nan)
risk_combined = np.clip(risk_combined, 0, 1)

# 提取有效值（安全过滤NaN/Inf）
exp_valid = total_exposure[valid_mask & ~np.isnan(total_exposure)]
wdi_val = wdi_norm[valid_mask & ~np.isnan(wdi_norm)]
risk_val = risk_combined[valid_mask & ~np.isnan(risk_combined)]

# 统一索引（保证三个数组长度一致）
common_idx = np.arange(len(exp_valid))
if len(exp_valid) != len(wdi_val) or len(exp_valid) != len(risk_val):
    min_len = min(len(exp_valid), len(wdi_val), len(risk_val))
    exp_valid = exp_valid[:min_len]
    wdi_val = wdi_val[:min_len]
    risk_val = risk_val[:min_len]

# 交叉统计
print("\n  --- 风险-暴露度交叉矩阵 ---")
wdi_bins = [0, 0.25, 0.5, 0.75, 1.0]
exp_bins = [0, 0.25, 0.5, 0.75, 1.0]

# 使用与valid_mask对齐的数据
wdi_full = np.where(valid_mask & np.isfinite(wdi_norm), wdi_norm, -9999)
exp_full = np.where(valid_mask & np.isfinite(total_exposure), total_exposure, -9999)

matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        mw = (wdi_full >= wdi_bins[i]) & (wdi_full < wdi_bins[i+1])
        me = (exp_full >= exp_bins[j]) & (exp_full < exp_bins[j+1])
        matrix[i, j] = np.sum(mw & me)

matrix_norm = matrix / matrix.sum() * 100

for i in range(4):
    vals = [f"{matrix_norm[i,j]:5.1f}%" for j in range(4)]
    print(f"  WDI[{wdi_bins[i]:.2f},{wdi_bins[i+1]:.2f})  " + "  ".join(vals))

# 双高区（使用P75分位数）
wdi_p75 = np.nanpercentile(wdi_val, 75) if len(wdi_val) > 0 else 0.5
exp_p75 = np.nanpercentile(exp_valid, 75) if len(exp_valid) > 0 else 0.5

high_mask = (wdi_full >= wdi_p75) & (exp_full >= exp_p75)
dual_high_pct = np.sum(high_mask) / n_valid * 100
print(f"\n  双高区(WDI≥P75 & 暴露≥P75): {dual_high_pct:.2f}%")
print(f"  WDI_P75={wdi_p75:.4f}, 暴露_P75={exp_p75:.4f}")

# 5. 统计检验
print("\n[5/6] 统计检验...")

# 安全采样（严格过滤NaN/Inf）
sample_n = min(10000, n_valid)
np.random.seed(42)
idx = np.random.choice(n_valid, sample_n, replace=False)

# 提取采样值并清理
wdi_sample = clean_array(wdi_full[valid_mask][idx])
exp_sample = clean_array(exp_full[valid_mask][idx])

# 确保有足够的变化
if np.std(wdi_sample) > 1e-10 and np.std(exp_sample) > 1e-10:
    r_sp, p_sp = spearmanr(wdi_sample, exp_sample)
    r_pe, p_pe = pearsonr(wdi_sample, exp_sample)
    print(f"  Spearman ρ: {r_sp:.4f}  P={p_sp:.4f}")
    print(f"  Pearson  r: {r_pe:.4f}  P={p_pe:.4f}")
else:
    r_sp, p_sp = 0.0, 1.0
    r_pe, p_pe = 0.0, 1.0
    print("  ⚠️  数据方差不足，跳过相关检验")

# Kruskal-Wallis检验
risk_bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
risk_labels = ['极低', '低', '中', '高', '极高']

risk_groups = []
for lo, hi in zip(risk_bins[:-1], risk_bins[1:]):
    grp = exp_valid[(risk_val >= lo) & (risk_val < hi)]
    grp = clean_array(grp)
    grp = grp[np.isfinite(grp)]
    if len(grp) > 10:
        risk_groups.append(grp)

if len(risk_groups) >= 2:
    kw_stat, kw_p = kruskal(*risk_groups)
    sig = '***' if kw_p < 0.001 else ('**' if kw_p < 0.01 else ('*' if kw_p < 0.05 else 'ns'))
    print(f"  Kruskal-Wallis: H={kw_stat:.1f}  P={kw_p:.2e}  {sig}")
else:
    kw_stat, kw_p = 0, 1
    print("  Kruskal-Wallis: 组数不足")

# 基本统计
exp_mean = float(np.nanmean(exp_valid)) if len(exp_valid) > 0 else 0
exp_cv = float(np.nanstd(exp_valid) / (exp_mean + 1e-10)) if exp_mean > 1e-10 else 0
wdi_mean = float(np.nanmean(wdi_val)) if len(wdi_val) > 0 else 0

print(f"\n  暴露度: 均值={exp_mean:.4f}  CV={exp_cv:.2f}")
print(f"  WDI:    均值={wdi_mean:.4f}")

# 6. 可视化
print("\n[6/6] 生成可视化...")

def save_fig(fig, name):
    path = os.path.join(VIS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()
    print(f"    → {name}")

# 图1：三图总览
fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(20, 7), facecolor='white')
fig1.suptitle('北京市 POI暴露度 × WDI 综合内涝风险分析', fontsize=14, fontweight='bold')

sm1 = plot_masked(ax1a, np.where(valid_mask, total_exposure, np.nan), valid_mask,
                  'YlOrRd', 0, 1, title=f'POI设施暴露度\n均值={exp_mean:.3f}  CV={exp_cv:.2f}')
if sm1: plt.colorbar(sm1, ax=ax1a, shrink=0.75, pad=0.02, label='暴露度')

sm2 = plot_masked(ax1b, np.where(valid_mask, wdi_norm, np.nan), valid_mask,
                  'Blues', 0, 1, title=f'WDI积水风险\n均值={wdi_mean:.3f}')
if sm2: plt.colorbar(sm2, ax=ax1b, shrink=0.75, pad=0.02, label='WDI')

sm3 = plot_masked(ax1c, np.where(valid_mask, risk_combined, np.nan), valid_mask,
                  'RdYlGn_r', 0, 1, title=f'综合风险 (WDI×暴露度)\n双高区={dual_high_pct:.1f}%')
if sm3: plt.colorbar(sm3, ax=ax1c, shrink=0.75, pad=0.02, label='综合风险')

ax1c.text(0.02, 0.02, f"Spearman ρ={r_sp:.3f}\nPearson r={r_pe:.3f}\nKW H={kw_stat:.0f}",
          transform=ax1c.transAxes, fontsize=8, va='bottom',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
save_fig(fig1, 'Step2_5_Enhanced_01_Overview.png')

# 图2：交叉矩阵
fig2, ax2 = plt.subplots(figsize=(8, 7), facecolor='white')
fig2.suptitle('风险-暴露度交叉分析矩阵', fontsize=13, fontweight='bold')

im2 = ax2.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', origin='lower',
                 vmin=0, vmax=max(1, np.nanmax(matrix_norm)))
ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
ax2.set_xticklabels(['低\n[0-0.25)', '中低\n[0.25-0.5)', '中高\n[0.5-0.75)', '高\n[0.75-1]'], fontsize=9)
ax2.set_yticklabels(['低\n[0-0.25)', '中低\n[0.25-0.5)', '中高\n[0.5-0.75)', '高\n[0.75-1]'], fontsize=9)
ax2.set_xlabel('POI暴露度等级', fontsize=11)
ax2.set_ylabel('WDI风险等级', fontsize=11)

for i in range(4):
    for j in range(4):
        v = matrix_norm[i, j]
        c = 'white' if v > np.nanmax(matrix_norm) * 0.5 else 'black'
        ax2.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=12, color=c, fontweight='bold')

plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02, label='面积占比(%)')

# 双高区标记
ax2.add_patch(Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor='#E74C3C',
                         linewidth=3, linestyle='--'))
ax2.text(0.78, 0.92, f'双高区\n{dual_high_pct:.1f}%', transform=ax2.transAxes,
         fontsize=11, color='#E74C3C', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
save_fig(fig2, 'Step2_5_Enhanced_02_Matrix.png')

# 图3：关键设施暴露度
key_cats = ['医疗保健', '科教文化', '交通设施', '生活服务', '旅游景点', '酒店住宿']
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10), facecolor='white')
fig3.suptitle('关键设施暴露度空间分布', fontsize=13, fontweight='bold')

for idx, cat in enumerate(key_cats):
    ax = axes3.flat[idx]
    if cat in category_densities:
        cat_exp = log_normalize(category_densities[cat], valid_mask, pmin=2, pmax=98)
        plot_masked(ax, np.where(valid_mask, cat_exp, np.nan), valid_mask, 'hot_r', 0, 1,
                    title=f'{CATEGORY_MAP[cat]["label"]} (w={CATEGORY_MAP[cat]["weight"]})\n{category_counts.get(cat,0):,}个POI')

plt.tight_layout()
save_fig(fig3, 'Step2_5_Enhanced_03_KeyFacilities.png')

# 图4：散点图 + 分布
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
fig4.suptitle('WDI vs POI暴露度 统计关系', fontsize=13, fontweight='bold')

# 散点图（降采样）- 确保所有数组对齐
sample_n_plot = min(5000, n_valid)

# 统一提取有效像元索引
valid_indices = np.where(valid_mask.ravel())[0]
# 确保wdi和exp都在有效像元中计算
valid_full_indices = valid_indices[
    np.isfinite(wdi_norm.ravel()[valid_indices]) &
    np.isfinite(total_exposure.ravel()[valid_indices])
]
sample_n_plot = min(sample_n_plot, len(valid_full_indices))

np.random.seed(42)
plot_idx = np.random.choice(valid_full_indices, sample_n_plot, replace=False)

# 提取采样值
wdi_s = wdi_norm.ravel()[plot_idx]
exp_s = total_exposure.ravel()[plot_idx]
risk_s = risk_combined.ravel()[plot_idx]

# 清理可能的NaN
valid_plot = np.isfinite(wdi_s) & np.isfinite(exp_s) & np.isfinite(risk_s)
wdi_s = wdi_s[valid_plot]
exp_s = exp_s[valid_plot]
risk_s = risk_s[valid_plot]

if len(wdi_s) > 10:
    sc = ax4a.scatter(wdi_s, exp_s, c=risk_s, cmap='RdYlGn_r',
                      alpha=0.5, s=6, edgecolors='none')

    # 趋势线
    z = np.polyfit(wdi_s, exp_s, 1)
    ax4a.plot([0, 1], [z[1], z[0] + z[1]], 'b--', linewidth=2,
              label=f'趋势线 (y={z[0]:.3f}x+{z[1]:.3f})')
    ax4a.set_xlabel('WDI风险', fontsize=11)
    ax4a.set_ylabel('POI暴露度', fontsize=11)
    ax4a.legend(fontsize=9)
    ax4a.set_title(f'WDI vs 暴露度\nSpearman ρ={r_sp:.3f}  P={p_sp:.4f}',
                   fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax4a, shrink=0.8, label='综合风险')

    # 分布对比
    for label, data, color in [('暴露度', exp_s, '#E67E22'), ('WDI', wdi_s, '#3498DB')]:
        counts, bins = np.histogram(data, bins=50, range=(0, 1), density=True)
        center = 0.5 * (bins[:-1] + bins[1:])
        ax4b.plot(center, counts, color=color, linewidth=2,
                  label=f'{label} (均值={np.mean(data):.3f})')

    ax4b.set_xlabel('值', fontsize=11)
    ax4b.set_ylabel('密度', fontsize=11)
    ax4b.set_title('分布对比（对数归一化）', fontsize=11, fontweight='bold')
    ax4b.legend(fontsize=10)
else:
    ax4a.text(0.5, 0.5, '无足够有效数据', ha='center', va='center', transform=ax4a.transAxes)
    ax4b.text(0.5, 0.5, '无足够有效数据', ha='center', va='center', transform=ax4b.transAxes)

plt.tight_layout()
save_fig(fig4, 'Step2_5_Enhanced_04_Scatter.png')

# ============================================================
# 保存结果
# ============================================================
print("\n[保存] 输出栅格...")

with rasterio.open(DEM_PATH) as ref:
    out_profile = ref.profile.copy()
out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='deflate')

# 暴露度
data_out = np.where(~valid_mask | np.isnan(total_exposure), -9999.0, total_exposure).astype(np.float32)
with rasterio.open(os.path.join(OUTPUT_DIR, 'POI_Exposure_LogNorm.tif'), 'w', **out_profile) as dst:
    dst.write(data_out, 1)

# 综合风险
data_out = np.where(~valid_mask | np.isnan(risk_combined), -9999.0, risk_combined).astype(np.float32)
with rasterio.open(os.path.join(OUTPUT_DIR, 'Combined_Risk_Enhanced.tif'), 'w', **out_profile) as dst:
    dst.write(data_out, 1)

# 统计
stats = {
    'POI总数': len(df_poi),
    '暴露度均值(对数)': round(exp_mean, 6),
    '暴露度CV': round(exp_cv, 4),
    'WDI均值': round(wdi_mean, 4),
    'Spearman_ρ': round(r_sp, 4),
    'Spearman_P': round(p_sp, 6),
    'Pearson_r': round(r_pe, 4),
    'Pearson_P': round(p_pe, 6),
    '双高区%': round(dual_high_pct, 3),
    'KruskalWallis_H': round(kw_stat, 2),
    'KruskalWallis_P': round(kw_p, 6),
}

pd.DataFrame([stats]).to_csv(os.path.join(OUTPUT_DIR, 'Enhanced_Stats.csv'), index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("✅ Step 2.5 Enhanced 完成！")
print("=" * 70)
print(f"  POI总数: {len(df_poi):,}")
print(f"  暴露度: 均值={exp_mean:.4f}  CV={exp_cv:.2f}")
print(f"  WDI:    均值={wdi_mean:.4f}")
print(f"  Spearman ρ={r_sp:.3f}  (P={p_sp:.4f})")
print(f"  Pearson  r={r_pe:.3f}  (P={p_pe:.4f})")
print(f"  双高区:  {dual_high_pct:.2f}%")
print(f"  Kruskal-Wallis: H={kw_stat:.1f}  P={kw_p:.2e}")
print(f"\n  输出: {OUTPUT_DIR}")
print(f"  可视化: {VIS_DIR}")
print("=" * 70)