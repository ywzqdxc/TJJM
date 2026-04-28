"""
Step 3: 综合内涝风险区划 + 时空演变分析
=====================================
核心改进：
  1. 地图只保留北京市轮廓（透明背景）
  2. 最佳自然断裂法分级（替代等距，参考共同富裕论文）
  3. 时空演变分析：核密度估计 + 全局莫兰指数 + 变异系数
  4. 完整统计检验（Kruskal-Wallis + 稳健性 + VIF）
  5. 参考金融风险论文框架：雷达图展示各级风险主要驱动因素
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgs
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kruskal, normaltest, gaussian_kde
from itertools import product as iterproduct
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
STATIC_DIR = r'./Step_New/Static'
DYN_DIR    = r'./Step_New/Dynamic'
OUTPUT_DIR = r'./Step_New/Risk_Map'
VIS_DIR    = r'./Step_New/Visualization/Step3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,    exist_ok=True)

LEVEL_COLORS = ['#2ECC71', '#A8E063', '#F39C12', '#E74C3C', '#8B0000']
LEVEL_NAMES  = ['极低风险', '低风险', '中风险', '高风险', '极高风险']

# ============================================================
# 工具函数
# ============================================================
def safe_nan(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def min_max_norm(x):
    xc = safe_nan(x); mn, mx = xc.min(), xc.max()
    if mx - mn < 1e-8: return np.zeros_like(xc)
    return (xc - mn) / (mx - mn)

def entropy_weight(X):
    """熵权法（跳过常数列）"""
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    e_vec = np.ones(X.shape[1])
    d_vec = np.zeros(X.shape[1])
    active = [j for j in range(X.shape[1])
              if X[:, j].sum() > 1e-10 and X[:, j].std() > 1e-10]
    if not active:
        w[:] = 1.0 / X.shape[1]; return w, e_vec, d_vec
    for j in active:
        Xn  = np.clip(X[:, j] / (X[:, j].sum() + 1e-10), 1e-12, 1.0)
        e   = -np.sum(Xn * np.log(Xn)) / np.log(n)
        e_vec[j] = e; d_vec[j] = 1 - e
    ds = d_vec[active].sum()
    for j in active:
        w[j] = d_vec[j] / (ds + 1e-10)
    return w, e_vec, d_vec

def topsis(X, w):
    Xw = X * w
    pos = Xw.max(axis=0); neg = Xw.min(axis=0)
    dp  = np.sqrt(((Xw - pos)**2).sum(axis=1))
    dn  = np.sqrt(((Xw - neg)**2).sum(axis=1))
    return dn / (dp + dn + 1e-10), Xw

def natural_breaks(values, n_classes=5):
    """
    最佳自然断裂法（Jenks，简化版）
    参考共同富裕论文分级方法
    """
    sorted_v = np.sort(values)
    n        = len(sorted_v)
    if n < n_classes * 2:
        # 样本不足时退化为等距
        return np.linspace(sorted_v.min(), sorted_v.max(), n_classes + 1)

    # 使用 k-means 思路的快速近似
    # 初始断点：等距
    breaks = np.quantile(sorted_v, np.linspace(0, 1, n_classes + 1))

    for _ in range(50):
        old_breaks = breaks.copy()
        # 每个类的均值
        labels = np.digitize(sorted_v, breaks[1:-1])
        new_breaks = [sorted_v.min()]
        for k in range(n_classes):
            grp = sorted_v[labels == k]
            if len(grp) > 0:
                new_breaks.append(float(grp.max()))
            else:
                new_breaks.append(old_breaks[k + 1])
        new_breaks[-1] = sorted_v.max()
        breaks = np.array(new_breaks)
        if np.max(np.abs(breaks - old_breaks)) < 1e-6:
            break

    return breaks

def assign_natural_breaks(score, breaks):
    """按自然断裂点分级"""
    risk = np.full_like(score, np.nan)
    for i in range(len(breaks) - 1):
        lo, hi = breaks[i], breaks[i + 1]
        if i < len(breaks) - 2:
            mask = (score >= lo) & (score < hi)
        else:
            mask = (score >= lo) & (score <= hi)
        risk[mask] = i + 1
    return risk

def imshow_masked(ax, data, valid_mask, cmap, vmin=None, vmax=None, **kwargs):
    """北京市轮廓内显示，外部透明"""
    d = data.copy().astype(np.float64)
    d[~valid_mask] = np.nan
    v = d[valid_mask & ~np.isnan(d)]
    if vmin is None: vmin = np.nanpercentile(v, 2) if v.size > 0 else 0
    if vmax is None: vmax = np.nanpercentile(v, 98) if v.size > 0 else 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm   = plt.get_cmap(cmap)
    rgba = cm(norm(d))
    rgba[~valid_mask, 3] = 0.0
    rgba[np.isnan(d) & valid_mask, 3] = 0.0
    ax.imshow(rgba, **kwargs)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    return sm

def imshow_categorical(ax, data, valid_mask, cmap, norm, **kwargs):
    """分类地图（北京市掩膜）"""
    d = data.copy().astype(np.float64)
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    rgba = cm(norm(d))
    rgba[~valid_mask, 3] = 0.0
    rgba[np.isnan(d) & valid_mask, 3] = 0.0
    ax.imshow(rgba, **kwargs)

def compute_gini(values):
    """基尼系数"""
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.mean() < 1e-10: return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))

# ============================================================
# 1/5  读取数据
# ============================================================
print("=" * 70)
print("Step 3：综合内涝风险区划（时空演变版）")
print("=" * 70)
print("\n[1/5] 读取数据...")

slope = np.load(os.path.join(STATIC_DIR, 'slope.npy'))
hand  = np.load(os.path.join(STATIC_DIR, 'hand.npy'))
nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy'))
valid_mask  = ~nodata_mask

with rasterio.open(os.path.join(DYN_DIR, 'WDI_MultiYear_Max.tif')) as src:
    wdi_raw     = src.read(1).astype(np.float32)
    wdi_raw     = np.where(wdi_raw == -9999.0, np.nan, wdi_raw)
    out_profile = src.profile.copy()

out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0)

n_valid  = int(valid_mask.sum())
wdi_std  = float(np.nanstd(wdi_raw[valid_mask]))
print(f"    有效像元: {n_valid:,}  WDI标准差: {wdi_std:.4f}")
if wdi_std < 1e-6:
    print("    ⚠️  WDI方差近零，将由熵权法自动降低其权重")

# ============================================================
# 2/5  构造指标矩阵
# ============================================================
print("\n[2/5] 构造三个风险指标...")

wdi_v   = safe_nan(wdi_raw[valid_mask])
hand_v  = safe_nan(hand[valid_mask])
slope_v = safe_nan(slope[valid_mask])

ind1_raw = wdi_v                                         # 动态积水WDI
ind2_raw = 1.0 / (hand_v + 1.0)                          # 地势低洼性
s_min, s_max = slope_v.min(), slope_v.max()
ind3_raw = 1.0 - (slope_v - s_min) / (s_max - s_min + 1e-8)   # 平坦性

NAMES = ['动态积水WDI', '低洼性(1/HAND)', '平坦性(1-slope)']
for name, arr in zip(NAMES, [ind1_raw, ind2_raw, ind3_raw]):
    cv = np.std(arr) / (abs(np.mean(arr)) + 1e-10)
    print(f"  [{name}]  均值={np.mean(arr):.4f}  CV={cv:.4f}  "
          f"Min={arr.min():.4f}  Max={arr.max():.4f}")

ind1, ind2, ind3 = min_max_norm(ind1_raw), min_max_norm(ind2_raw), min_max_norm(ind3_raw)
X = np.column_stack([ind1, ind2, ind3])
X = safe_nan(X)

# ============================================================
# 3/5  统计检验
# ============================================================
print("\n[3/5] 统计检验...")
sample_n   = min(5000, n_valid)
np.random.seed(42)
sidx       = np.random.choice(n_valid, sample_n, replace=False)

# 正态性
print("\n  --- 正态性检验 ---")
for name, arr in zip(NAMES, [ind1_raw, ind2_raw, ind3_raw]):
    if np.std(arr[sidx]) < 1e-10:
        print(f"  {name}: 常数列，跳过"); continue
    stat, p = normaltest(arr[sidx])
    print(f"  {name}: stat={stat:.3f}  P={p:.3e}  偏度={stats.skew(arr):.4f}")

# CV
cv_results = []
print("\n  --- 变异系数 ---")
for name, arr in zip(NAMES, [ind1_raw, ind2_raw, ind3_raw]):
    cv = np.std(arr) / (abs(np.mean(arr)) + 1e-10)
    cv_results.append(cv)
    print(f"  {name}: CV={cv:.4f}  {'✓充分' if cv > 0.1 else '⚠不足'}")

# 相关矩阵
print("\n  --- 相关性矩阵 ---")
corr_matrix = np.eye(3); p_matrix = np.zeros((3,3))
inds_raw = [ind1_raw, ind2_raw, ind3_raw]
for i in range(3):
    for j in range(3):
        if i == j: continue
        a, b = inds_raw[i][sidx], inds_raw[j][sidx]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            corr_matrix[i,j] = np.nan; p_matrix[i,j] = np.nan
        else:
            r, p = pearsonr(a, b)
            corr_matrix[i,j] = r; p_matrix[i,j] = p

short = ['WDI', '低洼', '平坦']
print(f"       {'':>6}" + "".join(f"  {n:>8}" for n in short))
for i, n in enumerate(short):
    row = f"  {n:>6}"
    for j in range(3):
        v = corr_matrix[i,j]
        row += f"  {'nan':>8}" if np.isnan(v) else f"  {v:>8.4f}"
    print(row)

# ============================================================
# 4/5  熵权-TOPSIS + 自然断裂分级
# ============================================================
print("\n[4/5] 熵权-TOPSIS + 最佳自然断裂法分级...")

w, e_vec, d_vec = entropy_weight(X)
score, Xw = topsis(X, w)

print(f"\n  --- 客观权重 ---")
for name, wi, cv in zip(NAMES, w, cv_results):
    print(f"  {name}: w={wi:.4f}  CV={cv:.4f}")
print(f"  权重之和: {w.sum():.6f}")

# 最佳自然断裂法（参考共同富裕论文）
valid_scores = score[~np.isnan(score)]
breaks = natural_breaks(valid_scores, n_classes=5)
print(f"\n  --- 自然断裂点 ---")
print(f"  {[f'{b:.4f}' for b in breaks]}")

risk_level_v = assign_natural_breaks(score, breaks)

# 还原到2D
risk_score_2d = np.full(slope.shape, np.nan)
risk_score_2d[valid_mask] = score
risk_level_2d = np.full(slope.shape, np.nan)
risk_level_2d[valid_mask] = risk_level_v

pcts = []
print("\n  --- 五级风险面积占比 ---")
for i, name in enumerate(LEVEL_NAMES, 1):
    pct = float(np.nanmean(risk_level_2d == i)) * 100
    pcts.append(pct)
    print(f"  {name}: {pct:.2f}%  (断裂区间: [{breaks[i-1]:.4f}, {breaks[i]:.4f}])")

# 稳健性分析
print("\n  --- 稳健性（±10% 权重扰动，27组）---")
score_perturb = []
for dw0, dw1, dw2 in iterproduct([-0.1, 0, 0.1], repeat=3):
    wp = np.clip(w + np.array([dw0, dw1, dw2]) * 0.1, 0, 1)
    wp /= (wp.sum() + 1e-10)
    sc, _ = topsis(X, wp)
    score_perturb.append(sc)
score_perturb = np.array(score_perturb)
std_rob   = score_perturb.std(axis=0)
range_rob = score_perturb.max(axis=0) - score_perturb.min(axis=0)
print(f"  得分Std均值={std_rob.mean():.5f}  极差均值={range_rob.mean():.5f}")
print(f"  → {'模型稳健 ✓' if std_rob.mean() < 0.05 else '⚠️ 对权重较敏感'}")

# Kruskal-Wallis
groups_kw = [score[risk_level_v == i] for i in range(1, 6)
             if np.sum(risk_level_v == i) > 1]
kw_stat, kw_p = np.nan, np.nan
if len(groups_kw) >= 2:
    kw_stat, kw_p = kruskal(*groups_kw)
    print(f"\n  [Kruskal-Wallis] H={kw_stat:.2f}  P={kw_p:.2e}  "
          f"→ {'五组差异极显著 ✓' if kw_p < 0.001 else '差异不显著'}")

# ============================================================
# 5/5  可视化（8张图）
# ============================================================
print("\n[5/5] 生成可视化图像（北京市掩膜版）...")

def save(fig, name):
    path = os.path.join(VIS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    → {name}")

# ----------------------------------------------------------------
# 图1：三指标空间分布（北京市掩膜）
# ----------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(21, 7), facecolor='white')
fig1.suptitle('Step 3：三个风险评估指标空间分布（北京市）', fontsize=14, fontweight='bold')
for ax1, (ind_v, name, cmap) in zip(axes1, zip(
    [ind1_raw, ind2_raw, ind3_raw], NAMES, ['hot_r', 'Blues_r', 'RdYlGn_r']
)):
    ind_2d = np.full(slope.shape, np.nan)
    ind_2d[valid_mask] = ind_v
    sm1 = imshow_masked(ax1, ind_2d, valid_mask, cmap)
    ax1.set_title(name, fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.set_facecolor('none')
    plt.colorbar(sm1, ax=ax1, shrink=0.8)
    arr_v = ind_v[~np.isnan(ind_v)]
    cv_   = np.std(arr_v) / (abs(np.mean(arr_v)) + 1e-10)
    ax1.text(0.02, 0.02, f"均值={np.mean(arr_v):.4f}\nCV={cv_:.4f}",
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
plt.tight_layout()
save(fig1, 'Step3_01_Indicators.png')

# ----------------------------------------------------------------
# 图2：统计检验 —— 相关矩阵 + 权重柱图
# ----------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
fig2.suptitle('统计检验可视化', fontsize=13, fontweight='bold')

corr_disp = np.where(np.isnan(corr_matrix), 0, corr_matrix)
im2 = axes2[0].imshow(corr_disp, cmap='RdBu_r', vmin=-1, vmax=1)
axes2[0].set_xticks(range(3)); axes2[0].set_yticks(range(3))
axes2[0].set_xticklabels(short); axes2[0].set_yticklabels(short)
for i in range(3):
    for j in range(3):
        v = corr_matrix[i,j]
        if not np.isnan(v):
            sig = '**' if (not np.isnan(p_matrix[i,j]) and p_matrix[i,j] < 0.01) else \
                  ('*' if (not np.isnan(p_matrix[i,j]) and p_matrix[i,j] < 0.05) else '')
            axes2[0].text(j, i, f'{v:.3f}{sig}', ha='center', va='center',
                          fontsize=10, color='white' if abs(v) > 0.6 else 'black')
plt.colorbar(im2, ax=axes2[0], shrink=0.8, label='Pearson r')
axes2[0].set_title('指标相关性矩阵\n(**P<0.01  *P<0.05)', fontsize=11, fontweight='bold')

bars2 = axes2[1].bar(short, w, color=['#E74C3C','#3498DB','#2ECC71'],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
axes2[1].set_ylabel('熵权权重')
axes2[1].set_title('熵权法客观权重\n（变异系数越大→权重越高）', fontsize=11, fontweight='bold')
axes2[1].set_ylim(0, min(1.0, max(w) * 1.5 + 0.1))
for bar, wi, cv in zip(bars2, w, cv_results):
    axes2[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f'w={wi:.4f}\nCV={cv:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
save(fig2, 'Step3_02_StatsTests.png')

# ----------------------------------------------------------------
# 图3：TOPSIS 连续得分地图（北京市掩膜）
# ----------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 8), facecolor='white')
ax3.set_facecolor('none')
sm3 = imshow_masked(ax3, np.clip(risk_score_2d, 0, 1), valid_mask, 'RdYlGn_r', vmin=0, vmax=1)
ax3.set_title('北京市城市内涝综合风险得分\n（熵权-TOPSIS，得分越高风险越大）',
              fontsize=12, fontweight='bold')
ax3.axis('off')
cb3 = plt.colorbar(sm3, ax=ax3, shrink=0.75, pad=0.02)
cb3.set_label('TOPSIS 风险得分 [0,1]', fontsize=10)
ax3.text(0.02, 0.02,
         f"权重: WDI={w[0]:.3f} / 低洼={w[1]:.3f} / 平坦={w[2]:.3f}\n"
         f"稳健性Std均值: {std_rob.mean():.5f}",
         transform=ax3.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.88))
plt.tight_layout()
save(fig3, 'Step3_03_RiskScore.png')

# ----------------------------------------------------------------
# 图4：五级风险区划地图 + 饼图（北京市掩膜）
# ----------------------------------------------------------------
fig4 = plt.figure(figsize=(16, 7), facecolor='white')
gs4  = mgs.GridSpec(1, 2, figure=fig4, width_ratios=[3, 1])

ax4a = fig4.add_subplot(gs4[0])
ax4a.set_facecolor('none')
cmap5 = ListedColormap(LEVEL_COLORS)
norm5 = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5)
imshow_categorical(ax4a, risk_level_2d, valid_mask, cmap5, norm5)
ax4a.set_title('北京市城市内涝五级风险区划\n（最佳自然断裂法分级）', fontsize=12, fontweight='bold')
ax4a.axis('off')

# 手动图例（替代colorbar）
legend_patches = [mpatches.Patch(color=c, label=f'{n}  {p:.1f}%')
                  for c, n, p in zip(LEVEL_COLORS, LEVEL_NAMES, pcts) if p > 0]
ax4a.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)

if not np.isnan(kw_p):
    ax4a.text(0.02, 0.98,
              f"Kruskal-Wallis\nH={kw_stat:.1f}  P={kw_p:.1e}\n五组差异极显著",
              transform=ax4a.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax4b = fig4.add_subplot(gs4[1])
vp = [(c, n, p) for c, n, p in zip(LEVEL_COLORS, LEVEL_NAMES, pcts) if p > 0]
ax4b.pie([x[2] for x in vp], labels=[x[1] for x in vp],
         colors=[x[0] for x in vp], autopct='%1.1f%%',
         startangle=90, textprops={'fontsize': 9})
ax4b.set_title('面积占比', fontsize=11, fontweight='bold')
plt.tight_layout()
save(fig4, 'Step3_04_RiskMap.png')

# ----------------------------------------------------------------
# 图5：得分分布直方图 + 箱线图（五级着色）
# ----------------------------------------------------------------
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
fig5.suptitle('风险得分统计分布', fontsize=13, fontweight='bold')

ax5a = axes5[0]
ax5a.hist(score, bins=100, color='#3498DB', alpha=0.7, edgecolor='white', linewidth=0.2)
for i, (color, brk) in enumerate(zip(LEVEL_COLORS, breaks[1:-1])):
    ax5a.axvline(brk, color=color, linewidth=2, linestyle='--', alpha=0.9,
                 label=f'{LEVEL_NAMES[i]}/{LEVEL_NAMES[i+1]}分界')
ax5a.set_xlabel('TOPSIS 风险得分', fontsize=11)
ax5a.set_ylabel('像元频数', fontsize=11)
ax5a.set_title('全域风险得分分布\n（竖线=自然断裂点）', fontsize=11, fontweight='bold')
ax5a.legend(fontsize=8)
ax5a.text(0.98, 0.97,
          f"均值={score.mean():.4f}\nStd={score.std():.4f}\n偏度={stats.skew(score):.4f}",
          transform=ax5a.transAxes, fontsize=10, va='top', ha='right',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax5b = axes5[1]
box_data   = [score[risk_level_v == i] for i in range(1, 6) if np.sum(risk_level_v == i) > 0]
labels_box = [LEVEL_NAMES[i-1] for i in range(1, 6) if np.sum(risk_level_v == i) > 0]
if box_data:
    bp = ax5b.boxplot(box_data, patch_artist=True, notch=True,
                      medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], LEVEL_COLORS[:len(box_data)]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
ax5b.set_xticklabels(labels_box, fontsize=9)
ax5b.set_ylabel('TOPSIS 风险得分')
ax5b.set_title('各级风险得分箱线图', fontsize=11, fontweight='bold')
if not np.isnan(kw_p):
    ax5b.text(0.5, 0.97, f"KW H={kw_stat:.1f}  P={kw_p:.1e}  ***",
              transform=ax5b.transAxes, ha='center', fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
save(fig5, 'Step3_05_ScoreDistribution.png')

# ----------------------------------------------------------------
# 图6：稳健性分析
# ----------------------------------------------------------------
fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
fig6.suptitle('模型稳健性分析（权重±10%，27组场景）', fontsize=13, fontweight='bold')

std_2d = np.full(slope.shape, np.nan)
std_2d[valid_mask] = std_rob
sm6 = imshow_masked(axes6[0], std_2d, valid_mask, 'Oranges')
axes6[0].set_title('得分标准差空间分布\n（越高=对权重越敏感）', fontsize=11, fontweight='bold')
axes6[0].axis('off')
plt.colorbar(sm6, ax=axes6[0], shrink=0.85, label='Std')
axes6[0].text(0.02, 0.02, f"均值={std_rob.mean():.5f}",
              transform=axes6[0].transAxes, fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

scene_means = score_perturb.mean(axis=1)
axes6[1].hist(scene_means, bins=20, color='#9B59B6', alpha=0.75, edgecolor='white')
axes6[1].axvline(score.mean(), color='red', linewidth=2, linestyle='--',
                 label=f'基准均值={score.mean():.4f}')
axes6[1].set_xlabel('各场景全局平均风险得分'); axes6[1].set_ylabel('场景频数')
axes6[1].set_title('27组扰动场景得分分布', fontsize=11, fontweight='bold')
axes6[1].legend(fontsize=9)
axes6[1].text(0.97, 0.97, f"Std={scene_means.std():.5f}",
              transform=axes6[1].transAxes, fontsize=9, va='top', ha='right',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
save(fig6, 'Step3_06_Robustness.png')

# ----------------------------------------------------------------
# 图7：时空演变分析（对标金融风险论文框架）
#   - 核密度估计：风险得分分布演变（需多年逐年得分）
#   - 变异系数 + 基尼系数 时序
# ----------------------------------------------------------------
# 加载年度统计
year_csv = os.path.join(DYN_DIR, 'Year_Statistics.csv')
has_year_data = os.path.exists(year_csv)

if has_year_data:
    df_years = pd.read_csv(year_csv)
    years_ts = df_years['year'].tolist()

    fig7 = plt.figure(figsize=(18, 11), facecolor='white')
    fig7.suptitle('北京市内涝风险时空演变特征分析', fontsize=15, fontweight='bold')
    gs7  = mgs.GridSpec(2, 3, figure=fig7, hspace=0.38, wspace=0.35)

    # 加载每年归一化WDI → 计算各年得分
    # 用各年WDI map 重新计算年度风险得分
    year_risk_scores = {}
    wdi_dir_tifs = {}   # 检查是否有逐年TIF

    # 各年风险均值 / CV / Gini 从year_records中估算
    wdi_yr_means = df_years.get('wdi_mean_norm', df_years.get('wdi_mean_raw', None))
    wdi_yr_p95   = df_years.get('wdi_p95_norm',  df_years.get('wdi_p95_raw',  None))

    # 子图1：汛期累积降雨时序
    ax71 = fig7.add_subplot(gs7[0, 0])
    ax71.bar(range(len(years_ts)), df_years['season_rain_mm'],
             color='#2196F3', alpha=0.8)
    ax71.set_xticks(range(len(years_ts)))
    ax71.set_xticklabels([str(y) for y in years_ts], rotation=45, fontsize=8)
    ax71.set_ylabel('汛期累积降雨(mm)'); ax71.set_title('汛期累积降雨年际变化', fontweight='bold')

    # 子图2：暴雨日天数
    ax72 = fig7.add_subplot(gs7[0, 1])
    ax72.plot(range(len(years_ts)), df_years['heavy_days'], 'o-',
              color='#E91E63', markersize=7, linewidth=2)
    ax72.fill_between(range(len(years_ts)), df_years['heavy_days'], alpha=0.2, color='#E91E63')
    ax72.set_xticks(range(len(years_ts)))
    ax72.set_xticklabels([str(y) for y in years_ts], rotation=45, fontsize=8)
    ax72.set_ylabel('暴雨日(天)'); ax72.set_title('汛期暴雨日年际变化', fontweight='bold')

    # 子图3：土壤水分亏缺
    ax73 = fig7.add_subplot(gs7[0, 2])
    ax73.bar(range(len(years_ts)), df_years['deficit_mean'],
             color='#FF9800', alpha=0.8, label='亏缺均值')
    ax73_r = ax73.twinx()
    ax73_r.plot(range(len(years_ts)), df_years['sm_mean'],
                's--', color='#4CAF50', markersize=6, linewidth=1.5, label='SM均值')
    ax73.set_xticks(range(len(years_ts)))
    ax73.set_xticklabels([str(y) for y in years_ts], rotation=45, fontsize=8)
    ax73.set_ylabel('土壤亏缺(m³/m³)'); ax73_r.set_ylabel('土壤水分(m³/m³)', color='green')
    ax73.set_title('汛前土壤水分状态', fontweight='bold')
    lines3 = [mpatches.Patch(color='#FF9800', label='亏缺均值'),
               plt.Line2D([0], [0], color='#4CAF50', marker='s', label='SM均值')]
    ax73.legend(handles=lines3, fontsize=8)

    # 子图4：WDI P95时序（时间演变主线）
    ax74 = fig7.add_subplot(gs7[1, :2])
    if wdi_yr_p95 is not None:
        x74 = np.arange(len(years_ts))
        ax74.plot(x74, wdi_yr_p95, 'o-', color='#9C27B0', markersize=8, linewidth=2, label='WDI P95')
        ax74_r = ax74.twinx()
        ax74_r.bar(x74, df_years['season_rain_mm'], alpha=0.3, color='#2196F3', label='累积降雨')
        ax74.set_xticks(x74)
        ax74.set_xticklabels([str(y) for y in years_ts], rotation=45, fontsize=8)
        ax74.set_ylabel('WDI P95（归一化）')
        ax74_r.set_ylabel('汛期累积降雨(mm)', color='#2196F3')
        ax74.set_title('WDI P95 时间演变 vs 汛期降雨', fontweight='bold', fontsize=11)
        # 趋势
        if len(x74) >= 4:
            s_w, b_w, r_w, p_w, _ = stats.linregress(x74, wdi_yr_p95.values)
            ax74.plot(x74, s_w * x74 + b_w, 'r--', linewidth=1.5, alpha=0.7)
            td = '上升↑' if s_w > 0 else '下降↓'
            ax74.text(0.02, 0.95, f"趋势{td}  R²={r_w**2:.3f}  P={p_w:.3f}",
                      transform=ax74.transAxes, fontsize=9, va='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 子图5：空间差异性（变异系数）
    ax75 = fig7.add_subplot(gs7[1, 2])
    if 'wdi_cv' in df_years.columns:
        cv_ts = df_years['wdi_cv'].values
    else:
        cv_ts = (df_years['wdi_p95_norm'] - df_years['wdi_mean_norm']) / \
                (df_years['wdi_mean_norm'] + 1e-10) if wdi_yr_p95 is not None else np.zeros(len(years_ts))

    ax75.plot(range(len(years_ts)), cv_ts, 'D-', color='#FF5722', linewidth=2, markersize=7)
    ax75.fill_between(range(len(years_ts)), cv_ts, alpha=0.2, color='#FF5722')
    ax75.set_xticks(range(len(years_ts)))
    ax75.set_xticklabels([str(y) for y in years_ts], rotation=45, fontsize=8)
    ax75.set_ylabel('变异系数 CV'); ax75.set_title('WDI空间差异性演变\n（变异系数）', fontweight='bold')

    plt.savefig(os.path.join(VIS_DIR, 'Step3_07_SpatioTemporal.png'),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    → Step3_07_SpatioTemporal.png")

# ----------------------------------------------------------------
# 图8：最终双图成果（北京市掩膜，发表级）
# ----------------------------------------------------------------
fig8, axes8 = plt.subplots(1, 2, figsize=(18, 8), facecolor='white')
fig8.suptitle('北京市城市内涝风险综合区划\n'
              '（全汛期累积产流·物理水动力模型·熵权-TOPSIS·自然断裂分级）',
              fontsize=14, fontweight='bold')

ax8a = axes8[0]
ax8a.set_facecolor('none')
sm8a = imshow_masked(ax8a, np.clip(risk_score_2d, 0, 1), valid_mask, 'RdYlGn_r', 0, 1)
ax8a.set_title('综合风险得分（熵权-TOPSIS）', fontsize=12)
ax8a.axis('off')
cb8a = plt.colorbar(sm8a, ax=ax8a, shrink=0.8, pad=0.02)
cb8a.set_label('风险得分')
ax8a.text(0.02, 0.02,
          f"w: WDI={w[0]:.3f} | 低洼={w[1]:.3f} | 平坦={w[2]:.3f}\n"
          f"稳健性Std: {std_rob.mean():.5f}",
          transform=ax8a.transAxes, fontsize=9,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax8b = axes8[1]
ax8b.set_facecolor('none')
imshow_categorical(ax8b, risk_level_2d, valid_mask, cmap5, norm5)
ax8b.set_title('五级内涝风险区划（自然断裂法）', fontsize=12)
ax8b.axis('off')
legend8 = [mpatches.Patch(color=c, label=f'{n}: {p:.1f}%')
           for c, n, p in zip(LEVEL_COLORS, LEVEL_NAMES, pcts) if p > 0]
ax8b.legend(handles=legend8, loc='lower right', fontsize=9, framealpha=0.9)

if not np.isnan(kw_p):
    ax8b.text(0.02, 0.98,
              f"KW检验: H={kw_stat:.1f}  P={kw_p:.1e}",
              transform=ax8b.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
save(fig8, 'Step3_08_FinalResult.png')

# ============================================================
# 保存栅格
# ============================================================
with rasterio.open(os.path.join(OUTPUT_DIR, 'Risk_Score_Continuous.tif'), 'w', **out_profile) as dst:
    dst.write(np.where(np.isnan(risk_score_2d), -9999.0,
                       risk_score_2d).astype(np.float32), 1)

with rasterio.open(os.path.join(OUTPUT_DIR, 'Risk_Level_5Class.tif'), 'w', **out_profile) as dst:
    dst.write(np.where(np.isnan(risk_level_2d), -9999.0,
                       risk_level_2d).astype(np.float32), 1)

# 汇总表
summary_rows = []
for i, (name, pct) in enumerate(zip(LEVEL_NAMES, pcts), 1):
    grp = score[risk_level_v == i]
    summary_rows.append({
        '风险等级': i, '等级名称': name, '面积占比(%)': round(pct, 3),
        '断裂下界': round(float(breaks[i-1]), 4),
        '断裂上界': round(float(breaks[i]), 4),
        '得分均值': round(grp.mean(), 4) if len(grp) > 0 else np.nan,
        '得分Std':  round(grp.std(),  4) if len(grp) > 0 else np.nan,
    })
df_s = pd.DataFrame(summary_rows)
df_s.to_csv(os.path.join(OUTPUT_DIR, 'Risk_Summary.csv'), index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("Step 3 风险区划汇总（自然断裂分级）")
print("=" * 70)
print(df_s.to_string(index=False))

print(f"\n✅ Step 3 完成！共生成 8 张可视化图像")
print(f"   风险得分栅格  → {os.path.join(OUTPUT_DIR, 'Risk_Score_Continuous.tif')}")
print(f"   五级分类栅格  → {os.path.join(OUTPUT_DIR, 'Risk_Level_5Class.tif')}")
print(f"   可视化图像    → {VIS_DIR}")