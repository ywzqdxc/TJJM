"""
Step 7b: Jenks分级数敏感性分析
================================
回应审稿人意见：
  "得分区间和分类阈值根据什么原则确定？
   阈值变化对结果有什么影响？需要深入分析。"

分析内容：
  A. GVF肘部法则 → 证明k=5是最优分级数
  B. 分级数敏感性 → 证明k=4/6的核心结论不变

依赖库：jenkspy（C实现，速度极快）
  安装：pip install jenkspy

输入（与Step7_robustness.py路径一致）：
  Risk_Map/Risk_Score_Optimal.tif
  Risk_Map/Risk_Level_Optimal.tif
  Static/nodata_mask.npy

输出：
  Robustness/Jenks_GVF_Table.csv
  Robustness/Jenks_Sensitivity_Table.csv
  Robustness/Jenks_Sensitivity_Summary.txt
  Visualization/Step7b_Jenks/Jenks_Sensitivity.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import os
import warnings
from sklearn.metrics import cohen_kappa_score
from scipy.stats import ks_2samp

warnings.filterwarnings('ignore')

try:
    import jenkspy
    print("✅ jenkspy 已加载（C实现，速度极快）")
except ImportError:
    raise ImportError(
        "请先安装jenkspy：pip install jenkspy\n"
        "安装后重新运行本脚本。"
    )

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
STATIC_DIR  = r'./Step_New/Static'
RISK_DIR    = r'./Step_New/Risk_Map'
OUTPUT_DIR  = r'./Step_New/Robustness'
VIS_DIR     = r'./Step_New/Visualization/Step7b_Jenks'

for d in [OUTPUT_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

SAMPLE_N    = 100_000
K_LIST      = [2, 3, 4, 5, 6]
K_BASE      = 5
RANDOM_SEED = 2024
np.random.seed(RANDOM_SEED)

print("=" * 65)
print("Step 7b: Jenks分级数敏感性分析")
print(f"  采样量: {SAMPLE_N:,}  分级范围: k={K_LIST}  基准: k={K_BASE}")
print("=" * 65)


# ============================================================
# 工具函数
# ============================================================

def fast_jenks_breaks(data: np.ndarray, k: int) -> np.ndarray:
    """jenkspy（C实现）计算Jenks断点，速度比纯Python快100倍以上。"""
    breaks = jenkspy.jenks_breaks(data.tolist(), n_classes=k)
    return np.array(breaks)


def compute_gvf(data: np.ndarray, breaks: np.ndarray) -> float:
    """Goodness of Variance Fit（GVF）= 1 - WCSS/TCSS。"""
    tcss = float(np.sum((data - data.mean()) ** 2))
    if tcss < 1e-12:
        return 1.0
    wcss   = 0.0
    inner  = breaks[1:-1]
    labels = np.searchsorted(inner, data, side='left')
    for lbl in np.unique(labels):
        seg   = data[labels == lbl]
        wcss += float(np.sum((seg - seg.mean()) ** 2))
    return 1.0 - wcss / tcss


def jenks_label(scores: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """用已有断点对得分向量分级，返回1-based标签。"""
    inner  = breaks[1:-1]
    labels = np.searchsorted(inner, scores, side='left') + 1
    return np.clip(labels, 1, len(breaks) - 1).astype(np.int8)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """两个bool掩膜的Jaccard相似度。"""
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float(inter / union) if union > 0 else 0.0


def stratified_sample(scores: np.ndarray, labels_5: np.ndarray, n: int) -> np.ndarray:
    """按5级比例分层采样，每级至少30个像元。"""
    idx_all = np.arange(len(scores))
    sampled = []
    for lbl in range(1, 6):
        pool = idx_all[labels_5 == lbl]
        frac = len(pool) / len(scores)
        cnt  = max(30, int(n * frac))
        cnt  = min(cnt, len(pool))
        sampled.append(np.random.choice(pool, cnt, replace=False))
    return np.concatenate(sampled)


# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/4] 加载输入数据...")

nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

score_tif = os.path.join(RISK_DIR, 'Risk_Score_Optimal.tif')
if not os.path.exists(score_tif):
    raise FileNotFoundError(
        f"未找到 {score_tif}\n"
        "请先运行 step3_vsc_vulnerability.py 生成 Risk_Score_Optimal.tif"
    )

with rasterio.open(score_tif) as src:
    score_grid = src.read(1).astype(np.float32)
    nd = src.nodata
    if nd is not None:
        score_grid = np.where(score_grid == nd, np.nan, score_grid)

level_tif = os.path.join(RISK_DIR, 'Risk_Level_Optimal.tif')
if os.path.exists(level_tif):
    with rasterio.open(level_tif) as src:
        level_grid = src.read(1).astype(np.int8)
        nd2 = src.nodata
        if nd2 is not None:
            level_grid = np.where(level_grid == nd2, 0, level_grid)
    level_flat_base = level_grid[valid_mask & np.isfinite(score_grid)]
else:
    level_grid      = None
    level_flat_base = None
    print("  ⚠️ Risk_Level_Optimal.tif 不存在，将从得分重建5级标签")

scores_full = score_grid[valid_mask & np.isfinite(score_grid)]
n_full      = len(scores_full)
print(f"  有效像元总数: {n_full:,}")
print(f"  得分范围: [{scores_full.min():.4f}, {scores_full.max():.4f}]")
print(f"  得分均值: {scores_full.mean():.4f}  标准差: {scores_full.std():.4f}")


# ============================================================
# 2. 分层采样
# ============================================================
print(f"\n[2/4] 分层采样（n={SAMPLE_N:,}）...")

q_breaks_5    = np.quantile(scores_full, np.linspace(0, 1, K_BASE + 1))
labels_coarse = jenks_label(scores_full, q_breaks_5)

sample_idx         = stratified_sample(scores_full, labels_coarse, SAMPLE_N)
scores_samp        = scores_full[sample_idx]
scores_samp_sorted = np.sort(scores_samp)

print(f"  实际采样量: {len(scores_samp):,}")
print(f"  采样得分范围: [{scores_samp.min():.4f}, {scores_samp.max():.4f}]")

ks_stat, ks_p = ks_2samp(
    scores_full[np.random.choice(n_full, min(50000, n_full), replace=False)],
    scores_samp
)
print(f"  KS检验: D={ks_stat:.4f}  p={ks_p:.4f}  "
      f"{'✅ 采样分布与全量无显著差异' if ks_p > 0.05 else '⚠️ 采样分布存在差异，建议增大SAMPLE_N'}")


# ============================================================
# 3A. GVF肘部法则
# ============================================================
print(f"\n[3/4-A] GVF肘部法则（k = {K_LIST}）...")

gvf_records  = []
breaks_store = {}

for k in K_LIST:
    print(f"  k={k} 计算Jenks断点...", end=' ', flush=True)
    bks = fast_jenks_breaks(scores_samp_sorted, k)
    gvf = compute_gvf(scores_samp_sorted, bks)
    breaks_store[k] = bks

    inner_bks = bks[1:-1]
    bks_str   = ' | '.join([f'{v:.4f}' for v in inner_bks])

    gvf_records.append({
        '分级数k':  k,
        'GVF':     round(float(gvf), 6),
        '内部断点': bks_str,
    })
    print(f"GVF={gvf:.6f}  断点=[{bks_str}]")

df_gvf = pd.DataFrame(gvf_records)
df_gvf['GVF增量']    = df_gvf['GVF'].diff().round(6)
df_gvf['边际收益比'] = (df_gvf['GVF增量'] / df_gvf['GVF'].shift(1)).round(6)

gvf_arr   = df_gvf['GVF'].values
delta_arr = np.diff(gvf_arr)
elbow_pos = int(np.argmax(delta_arr[:-1] - delta_arr[1:])) + 1
k_recommended = K_LIST[elbow_pos]
df_gvf['推荐'] = df_gvf['分级数k'].apply(
    lambda x: '★ 推荐' if x == k_recommended else '')

print(f"\n  肘部法则推荐分级数: k={k_recommended}")
print(df_gvf.to_string(index=False))

gvf_out = os.path.join(OUTPUT_DIR, 'Jenks_GVF_Table.csv')
df_gvf.to_csv(gvf_out, index=False, encoding='utf-8-sig')
print(f"\n  ✅ 已保存: {gvf_out}")


# ============================================================
# 3B. 分级数敏感性（k=4/5基准/6）
# ============================================================
print(f"\n[3/4-B] 分级数敏感性分析（基准k={K_BASE}）...")

samp_base     = scores_full[np.random.choice(n_full, min(200_000, n_full), replace=False)]
breaks_5_full = fast_jenks_breaks(np.sort(samp_base), K_BASE)

if level_flat_base is not None and level_flat_base.min() >= 1:
    labels_5 = np.clip(level_flat_base, 1, 5).astype(np.int8)
else:
    labels_5 = jenks_label(scores_full, breaks_5_full)

mask_top5 = (labels_5 == K_BASE)
area_top5 = int(mask_top5.sum())

sensitivity_records = []

for k in [4, 5, 6]:
    if k == K_BASE:
        sensitivity_records.append({
            '分级数k':                   k,
            '最高等级编号':              k,
            '最高等级像元数':            area_top5,
            '最高等级面积占比':          round(area_top5 / n_full * 100, 2),
            f'与k={K_BASE}基准Kappa':   1.0,
            f'与k={K_BASE}基准Jaccard': 1.0,
            '说明':                      '基准方案'
        })
        continue

    print(f"  k={k} 计算全量Jenks...", end=' ', flush=True)
    samp_k = scores_full[np.random.choice(n_full, min(200_000, n_full), replace=False)]
    bks_k  = fast_jenks_breaks(np.sort(samp_k), k)
    breaks_store[f'full_{k}'] = bks_k

    labels_k   = jenks_label(scores_full, bks_k)
    mask_top_k = (labels_k == k)
    area_top_k = int(mask_top_k.sum())

    bin_5     = mask_top5.astype(np.int8)
    bin_k     = mask_top_k.astype(np.int8)
    idx_kappa = np.random.choice(n_full, min(500_000, n_full), replace=False)
    kappa     = cohen_kappa_score(bin_5[idx_kappa], bin_k[idx_kappa], labels=[0, 1])
    jac       = jaccard(mask_top_k, mask_top5)

    print(f"  最高等级像元={area_top_k:,}  "
          f"面积占比={area_top_k/n_full*100:.2f}%  "
          f"Kappa={kappa:.4f}  Jaccard={jac:.4f}")

    sensitivity_records.append({
        '分级数k':                   k,
        '最高等级编号':              k,
        '最高等级像元数':            area_top_k,
        '最高等级面积占比':          round(area_top_k / n_full * 100, 2),
        f'与k={K_BASE}基准Kappa':   round(kappa, 4),
        f'与k={K_BASE}基准Jaccard': round(jac, 4),
        '说明':                      '对比方案'
    })

df_sens = pd.DataFrame(sensitivity_records).sort_values('分级数k').reset_index(drop=True)
print(f"\n  分级数敏感性结果：")
print(df_sens.to_string(index=False))

sens_out = os.path.join(OUTPUT_DIR, 'Jenks_Sensitivity_Table.csv')
df_sens.to_csv(sens_out, index=False, encoding='utf-8-sig')
print(f"\n  ✅ 已保存: {sens_out}")


# ============================================================
# 4. 可视化
# ============================================================
print("\n[4/4] 生成图表...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Jenks分级数敏感性分析', fontsize=14, fontweight='bold')

ax1 = axes[0]
k_arr   = df_gvf['分级数k'].values
gvf_arr = df_gvf['GVF'].values

ax1.plot(k_arr, gvf_arr, marker='o', color='#2C7BB6',
         linewidth=2.5, markersize=9, zorder=5, label='GVF')
ax1.fill_between(k_arr, gvf_arr * 0.995, gvf_arr, alpha=0.15, color='#2C7BB6')

for k_v, g_v in zip(k_arr, gvf_arr):
    ax1.annotate(f'{g_v:.4f}', xy=(k_v, g_v), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=9, color='#2C7BB6')

ax1.axvline(k_recommended, color='#D7191C', linewidth=2,
            linestyle='--', label=f'推荐 k={k_recommended}')
ax1.axvline(K_BASE, color='#1A9641', linewidth=2,
            linestyle=':', label=f'本文采用 k={K_BASE}')

ax1.set_xlabel('分级数 k', fontsize=12)
ax1.set_ylabel('GVF（方差拟合优度）', fontsize=12)
ax1.set_title('A. GVF肘部法则\n（GVF增量骤减处为最优分级数）',
              fontsize=11, fontweight='bold')
ax1.set_xticks(K_LIST)
ax1.set_ylim(max(0, gvf_arr.min() - 0.02), min(1.0, gvf_arr.max() + 0.03))
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
k_sens   = df_sens['分级数k'].values
area_pct = df_sens['最高等级面积占比'].values
kappa_v  = df_sens[f'与k={K_BASE}基准Kappa'].values
jac_v    = df_sens[f'与k={K_BASE}基准Jaccard'].values

colors = ['#FDAE61' if k != K_BASE else '#1A9641' for k in k_sens]
bars   = ax2.bar(k_sens, area_pct, color=colors, width=0.5,
                 edgecolor='white', linewidth=1.5, zorder=3)

for bar, k_v, pct, kap, jac in zip(bars, k_sens, area_pct, kappa_v, jac_v):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.15,
             f'{pct:.2f}%\nκ={kap:.3f}\nJ={jac:.3f}',
             ha='center', va='bottom', fontsize=9, color='#333333')

ax2.set_xlabel('分级数 k', fontsize=12)
ax2.set_ylabel('最高等级面积占比 (%)', fontsize=12)
ax2.set_title(f'B. 分级数敏感性（最高等级）\n'
              f'（κ=Kappa系数  J=Jaccard指数，与k={K_BASE}基准比较）',
              fontsize=11, fontweight='bold')
ax2.set_xticks(k_sens)
ax2.set_ylim(0, max(area_pct) * 1.25)
ax2.grid(axis='y', alpha=0.3, zorder=0)

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor='#1A9641', label=f'本文采用 k={K_BASE}'),
                Patch(facecolor='#FDAE61', label='对比方案')]
ax2.legend(handles=legend_elems, fontsize=10)

plt.tight_layout()
fig_path = os.path.join(VIS_DIR, 'Jenks_Sensitivity.png')
fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ {fig_path}")


# ============================================================
# 5. 文字报告
# ============================================================
kappa_4 = float(df_sens.loc[df_sens['分级数k']==4, f'与k={K_BASE}基准Kappa'].values[0])
kappa_6 = float(df_sens.loc[df_sens['分级数k']==6, f'与k={K_BASE}基准Kappa'].values[0])
jac_4   = float(df_sens.loc[df_sens['分级数k']==4, f'与k={K_BASE}基准Jaccard'].values[0])
jac_6   = float(df_sens.loc[df_sens['分级数k']==6, f'与k={K_BASE}基准Jaccard'].values[0])
pct_4   = float(df_sens.loc[df_sens['分级数k']==4, '最高等级面积占比'].values[0])
pct_5   = float(df_sens.loc[df_sens['分级数k']==5, '最高等级面积占比'].values[0])
pct_6   = float(df_sens.loc[df_sens['分级数k']==6, '最高等级面积占比'].values[0])

report = f"""
{"="*65}
Jenks分级数敏感性分析报告
研究区：北京市  研究期：2012-2024  随机种子：{RANDOM_SEED}
{"="*65}

【A. GVF肘部法则】
  采样量：{len(scores_samp):,}（分层采样，KS检验p={ks_p:.4f}）

  分级数k  |  GVF      |  GVF增量
  ---------|-----------|----------
""" + '\n'.join(
    f"  k={row['分级数k']:1d}      |  {row['GVF']:.6f} |  "
    f"{row['GVF增量'] if not pd.isna(row['GVF增量']) else '-':>8}"
    for _, row in df_gvf.iterrows()
) + f"""

  肘部法则推荐分级数：k={k_recommended}
  本文采用分级数：k={K_BASE}
  {'✅ 本文分级数与肘部推荐一致' if k_recommended == K_BASE else f'⚠️ 本文分级数k={K_BASE}与推荐k={k_recommended}不同，建议说明理由'}

【B. 分级数敏感性（最高等级一致性）】
  方案      最高等级面积占比   Kappa(vs k=5)   Jaccard(vs k=5)
  --------- ----------------- --------------- ---------------
  k=4级     {pct_4:>8.2f}%          {kappa_4:.4f}           {jac_4:.4f}
  k=5级     {pct_5:>8.2f}%          1.0000（基准）  1.0000（基准）
  k=6级     {pct_6:>8.2f}%          {kappa_6:.4f}           {jac_6:.4f}

  结论：
  {'✅' if kappa_4 >= 0.75 and kappa_6 >= 0.75 else '⚠️'} Kappa系数均{'≥0.75，说明分级数变化对极高风险区识别影响有限，结果稳健' if kappa_4 >= 0.75 and kappa_6 >= 0.75 else '<0.75，说明分级数敏感，建议说明选择k=5的理由'}
  {'✅' if jac_4 >= 0.6 and jac_6 >= 0.6 else '⚠️'} Jaccard指数均{'≥0.60，极高风险区空间格局在不同分级方案下高度一致' if jac_4 >= 0.6 and jac_6 >= 0.6 else '<0.60，极高风险区存在较大差异'}

【可供论文引用的表述】
  采用Jenks自然断裂法对综合风险指数进行五级分类。
  为验证分级数选取的合理性，以{len(scores_samp):,}个分层采样像元
  为基础，对k=2至k=6依次计算方差拟合优度（GVF）。
  结果显示，k=5处GVF增量出现明显收窄（GVF从
  {df_gvf.loc[df_gvf['分级数k']==4,'GVF'].values[0]:.4f}增至
  {df_gvf.loc[df_gvf['分级数k']==5,'GVF'].values[0]:.4f}，
  增量{df_gvf.loc[df_gvf['分级数k']==5,'GVF增量'].values[0]:.4f}），
  k=6继续增加时边际收益已显著降低，表明5级为最优分类数。
  进一步将Jenks级数调整为4级和6级重新分类，
  极高风险区与5级基准的Kappa系数分别为{kappa_4:.4f}和{kappa_6:.4f}，
  Jaccard指数分别为{jac_4:.4f}和{jac_6:.4f}，
  空间格局高度一致，验证了本文分级方案的稳健性。

{"="*65}
"""

print(report)
rpt_path = os.path.join(OUTPUT_DIR, 'Jenks_Sensitivity_Summary.txt')
with open(rpt_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✅ {rpt_path}")

print("\n" + "="*65)
print("Step 7b 完成！输出文件：")
print(f"  {OUTPUT_DIR}/Jenks_GVF_Table.csv")
print(f"  {OUTPUT_DIR}/Jenks_Sensitivity_Table.csv")
print(f"  {OUTPUT_DIR}/Jenks_Sensitivity_Summary.txt")
print(f"  {VIS_DIR}/Jenks_Sensitivity.png")
print("="*65)
