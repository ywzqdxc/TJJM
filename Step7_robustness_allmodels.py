"""
Step 7 补充脚本：WLC / TOPSIS / VSC 三模型稳健性对比验证
=========================================================
A. 权重扰动测试（Weight Perturbation）
   对组合权重施加 N=100 次 ±10% 随机扰动，
   分别计算三模型扰动得分与原始得分的 Spearman ρ。
   结论：均值 ≥ 0.95 → 权重选择稳健

B. Bootstrap 重抽样稳定性验证
   从有效像元有放回抽取 500,000 个样本，重复 N=100 次，
   分别计算三模型 Cohen's Kappa 系数。
   结论：均值 ≥ 0.80 → 评估结果稳健

输出：
  Robustness/Weight_Perturbation_AllModels.csv
  Robustness/Bootstrap_AllModels.csv
  Visualization/Step7_Robustness/Robustness_AllModels_Report.png
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2024)

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
OUTPUT_DIR = r'./Step_New/Robustness'
VIS_DIR    = r'./Step_New/Visualization/Step7_Robustness'
for d in [OUTPUT_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

# 超参数
N_PERTURB   = 100
PERTURB_PCT = 0.10
N_BOOT      = 100
BOOT_SAMPLE = 500_000
KAPPA_GOOD  = 0.80
RHO_GOOD    = 0.95

MODEL_NAMES  = ['WLC', 'TOPSIS', 'VSC']
MODEL_COLORS = {'WLC': '#2C7BB6', 'TOPSIS': '#D7191C', 'VSC': '#1A9641'}

IND_LABELS = ['降雨量R', '径流CR', 'TWI', 'HAND', '积水点WP',
              '人口PD', '路网RD', '避难SH', '医院HO', '消防FS']

print("=" * 70)
print("Step 7 补充：WLC / TOPSIS / VSC 三模型稳健性对比验证")
print(f"A. 权重扰动: N={N_PERTURB}次, ±{PERTURB_PCT*100:.0f}%  阈值ρ≥{RHO_GOOD}")
print(f"B. Bootstrap: N={N_BOOT}次, {BOOT_SAMPLE:,}像元/次  阈值κ≥{KAPPA_GOOD}")
print("=" * 70)


# ============================================================
# 工具函数
# ============================================================

def load_npy(path):
    if not os.path.exists(path):
        print(f"  缺失: {path}")
        return None
    return np.load(path).astype(np.float32)


def load_tif(path, h=None, w=None, dst_crs=None, dst_transform=None):
    if not os.path.exists(path):
        print(f"  缺失: {path}")
        return None
    with rasterio.open(path) as src:
        if h is not None and (src.height != h or src.width != w):
            arr = np.full((h, w), -9999.0, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=arr,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, dst_nodata=-9999.0)
        else:
            arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
    return np.where(arr < -1e10, np.nan, arr)


def minmax_norm(arr, valid_mask, direction='positive'):
    result = np.full_like(arr, np.nan, dtype=np.float32)
    vals   = arr[valid_mask & np.isfinite(arr)]
    if vals.size < 10:
        return result
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-10:
        result[valid_mask & np.isfinite(arr)] = 0.5
        return result
    normed = np.clip((arr - mn) / (mx - mn), 0.0, 1.0)
    if direction == 'negative':
        normed = 1.0 - normed
    result[valid_mask & np.isfinite(arr)] = normed[valid_mask & np.isfinite(arr)]
    return result


def jenks_breaks(vals, k=5, n_sample=80000, n_iter=100):
    v = vals[np.isfinite(vals)]
    if v.size > n_sample:
        v = np.random.choice(v, n_sample, replace=False)
    sv  = np.sort(v)
    bks = np.quantile(sv, np.linspace(0, 1, k + 1))
    for _ in range(n_iter):
        old    = bks.copy()
        labels = np.digitize(sv, bks[1:-1])
        new    = [sv.min()]
        for ki in range(k):
            grp = sv[labels == ki]
            new.append(float(grp.max()) if len(grp) > 0 else old[ki + 1])
        new[-1] = float(sv.max())
        bks = np.array(new)
        if np.max(np.abs(bks - old)) < 1e-7:
            break
    return bks


def jenks_label(scores, breaks):
    inner  = breaks[1:-1]
    labels = np.searchsorted(inner, scores, side='left') + 1
    return np.clip(labels, 1, 5).astype(np.int8)


def compute_score(model, X, w):
    """根据模型名称计算得分，w 为当前权重向量"""
    if model == 'WLC':
        return (X @ w).astype(np.float32)

    elif model == 'TOPSIS':
        Xw    = X * w[np.newaxis, :]
        Z_pos = Xw.max(axis=0)
        Z_neg = Xw.min(axis=0)
        D_pos = np.sqrt(((Xw - Z_pos) ** 2).sum(axis=1))
        D_neg = np.sqrt(((Xw - Z_neg) ** 2).sum(axis=1))
        return (D_neg / (D_pos + D_neg + 1e-10)).astype(np.float32)

    elif model == 'VSC':
        w_E = w[0:5] / (w[0:5].sum() + 1e-12)
        w_S = w[5:7] / (w[5:7].sum() + 1e-12)
        w_C = w[7:10] / (w[7:10].sum() + 1e-12)
        E_f = X[:, 0:5] @ w_E
        S_f = X[:, 5:7] @ w_S
        C_f = X[:, 7:10] @ w_C
        raw = E_f + S_f + C_f
        mn, mx = raw.min(), raw.max()
        return ((raw - mn) / (mx - mn + 1e-10)).astype(np.float32)

    else:
        raise ValueError(f"未知模型: {model}")


# ============================================================
# 一、加载基准数据
# ============================================================
print("\n[1/4] 加载基准数据...")

with rasterio.open(DEM_PATH) as ref:
    h, w          = ref.height, ref.width
    dst_crs       = ref.crs
    dst_transform = ref.transform

nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

# 读取组合权重
weights_csv = os.path.join(RISK_DIR, 'VSC_Weights.csv')
if not os.path.exists(weights_csv):
    raise FileNotFoundError(f"VSC_Weights.csv 未找到，请先运行 step3_vsc_vulnerability.py")
df_w    = pd.read_csv(weights_csv)
w_combo = df_w['组合'].values.astype(np.float64)
m_ind   = len(w_combo)
print(f"  组合权重（{m_ind}个指标）: {[f'{v:.4f}' for v in w_combo]}")

# 加载10个原始指标
print("\n  加载10个原始指标...")
rain_raw = load_npy(os.path.join(DYN_DIR,    'Precipitation_Mean.npy'))
cr_raw   = load_npy(os.path.join(DYN_DIR,    'CR_MultiYear_Mean.npy'))
twi_raw  = load_npy(os.path.join(STATIC_DIR, 'twi.npy'))
hand_raw = load_npy(os.path.join(STATIC_DIR, 'hand.npy'))
wp_raw   = load_tif(os.path.join(EXT_DIR, 'waterlogging_point_density_30m.tif'),
                    h, w, dst_crs, dst_transform)
pop_raw  = load_tif(os.path.join(EXT_DIR, 'population_density_30m.tif'),
                    h, w, dst_crs, dst_transform)
road_raw = load_tif(os.path.join(EXT_DIR, 'road_density_30m.tif'),
                    h, w, dst_crs, dst_transform)
sh_raw   = load_tif(os.path.join(EXT_DIR, 'shelter_density_30m.tif'),
                    h, w, dst_crs, dst_transform)
ho_raw   = load_tif(os.path.join(EXT_DIR, 'hospital_density_30m.tif'),
                    h, w, dst_crs, dst_transform)
fs_raw   = load_tif(os.path.join(EXT_DIR, 'firestation_density_30m.tif'),
                    h, w, dst_crs, dst_transform)

raw_list   = [rain_raw, cr_raw, twi_raw, hand_raw, wp_raw,
              pop_raw, road_raw, sh_raw, ho_raw, fs_raw]
directions = ['positive', 'positive', 'positive', 'negative', 'positive',
              'positive', 'positive', 'negative', 'negative', 'negative']

# Min-Max 归一化
print("\n  Min-Max 归一化...")
norm_list = []
for lab, arr, dire in zip(IND_LABELS, raw_list, directions):
    if arr is None:
        norm = np.where(valid_mask, 0.0, np.nan).astype(np.float32)
        print(f"  {lab}: 缺失，用0填充")
    else:
        norm = minmax_norm(arr, valid_mask, dire)
        v    = norm[valid_mask & np.isfinite(norm)]
        print(f"  {lab:10s} ({dire[:3]}): 均值={v.mean():.4f}")
    norm_list.append(norm)

# 构建完整有效像元掩膜
all_valid_px = valid_mask.copy()
for arr in norm_list:
    all_valid_px &= np.isfinite(arr)

n_px   = int(all_valid_px.sum())
X_flat = np.column_stack([arr[all_valid_px] for arr in norm_list]).astype(np.float64)
X_flat = np.clip(X_flat, 0.0, 1.0)
print(f"\n  X_flat: {X_flat.shape}  完整有效像元={n_px:,}")

# 计算三模型基准得分
print("\n  计算三模型基准得分...")
s_base_dict = {}
for model in MODEL_NAMES:
    s = compute_score(model, X_flat, w_combo)
    s_base_dict[model] = s
    print(f"  {model:8s}: 均值={s.mean():.4f}  Std={s.std():.4f}  "
          f"范围=[{s.min():.4f},{s.max():.4f}]")

# 计算各模型 Jenks 断点
print("\n  计算各模型 Jenks 断点...")
breaks_dict = {}
for model in MODEL_NAMES:
    bks = jenks_breaks(s_base_dict[model])
    breaks_dict[model] = bks
    print(f"  {model:8s}: {[f'{b:.4f}' for b in bks]}")

# 加载原始等级（Bootstrap 基准）
level_tif = os.path.join(RISK_DIR, 'Risk_Level_Optimal.tif')
level_grid = load_tif(level_tif)
if level_grid is not None:
    level_flat = level_grid[all_valid_px].astype(np.int8)
    level_flat = np.clip(level_flat, 1, 5)
    print(f"\n  原始等级（Risk_Level_Optimal.tif）: "
          f"{dict(zip(*np.unique(level_flat, return_counts=True)))}")
else:
    print("\n  Risk_Level_Optimal.tif 不存在，用 TOPSIS 基准得分重算等级")
    level_flat = jenks_label(s_base_dict['TOPSIS'], breaks_dict['TOPSIS'])


# ============================================================
# 二、权重扰动测试（三模型各 N=100 次）
# ============================================================
print(f"\n[2/4] 权重扰动测试（N={N_PERTURB}, ±{PERTURB_PCT*100:.0f}%）...")

perturb_results = {model: [] for model in MODEL_NAMES}

for model in MODEL_NAMES:
    print(f"\n  --- {model} ---")
    s_base = s_base_dict[model]
    for i in range(N_PERTURB):
        # δ ~ U(-0.1, 0.1)，w_perturb = w_combo × (1 + δ)
        delta     = np.random.uniform(-PERTURB_PCT, PERTURB_PCT, size=m_ind)
        w_perturb = w_combo * (1.0 + delta)
        w_perturb = np.clip(w_perturb, 0.0, None)
        w_perturb = w_perturb / (w_perturb.sum() + 1e-12)

        s_p      = compute_score(model, X_flat, w_perturb)
        rho, _   = spearmanr(s_base, s_p)
        perturb_results[model].append(float(rho))

        if (i + 1) % 25 == 0:
            cur_mean = np.mean(perturb_results[model])
            print(f"    进度: {i+1}/{N_PERTURB}  当前ρ均值={cur_mean:.6f}")

# 汇总权重扰动统计
perturb_stats = {}
for model in MODEL_NAMES:
    rhos = perturb_results[model]
    perturb_stats[model] = {
        'mean': float(np.mean(rhos)),
        'std':  float(np.std(rhos)),
        'min':  float(np.min(rhos)),
        'pass': float(np.mean(rhos)) >= RHO_GOOD,
    }

print(f"\n  权重扰动测试汇总（阈值 ρ ≥ {RHO_GOOD}）:")
print(f"  {'模型':<10} {'ρ均值':>10} {'ρ标准差':>10} {'ρ最小值':>10} {'结论':>8}")
print("  " + "-" * 52)
for model in MODEL_NAMES:
    st = perturb_stats[model]
    flag = "通过" if st['pass'] else "未通过"
    print(f"  {model:<10} {st['mean']:>10.6f} {st['std']:>10.6f} "
          f"{st['min']:>10.6f} {flag:>8}")

# 保存权重扰动 CSV
rows_p = {
    '扰动次序': list(range(1, N_PERTURB + 1)),
    'WLC_rho':    perturb_results['WLC'],
    'TOPSIS_rho': perturb_results['TOPSIS'],
    'VSC_rho':    perturb_results['VSC'],
}
df_perturb = pd.DataFrame(rows_p)
summary_rows = pd.DataFrame([
    {'扰动次序': '汇总_均值',
     'WLC_rho':    perturb_stats['WLC']['mean'],
     'TOPSIS_rho': perturb_stats['TOPSIS']['mean'],
     'VSC_rho':    perturb_stats['VSC']['mean']},
    {'扰动次序': '汇总_标准差',
     'WLC_rho':    perturb_stats['WLC']['std'],
     'TOPSIS_rho': perturb_stats['TOPSIS']['std'],
     'VSC_rho':    perturb_stats['VSC']['std']},
    {'扰动次序': '汇总_最小值',
     'WLC_rho':    perturb_stats['WLC']['min'],
     'TOPSIS_rho': perturb_stats['TOPSIS']['min'],
     'VSC_rho':    perturb_stats['VSC']['min']},
])
df_perturb = pd.concat([df_perturb, summary_rows], ignore_index=True)
out_p = os.path.join(OUTPUT_DIR, 'Weight_Perturbation_AllModels.csv')
df_perturb.to_csv(out_p, index=False, encoding='utf-8-sig')
print(f"\n  Weight_Perturbation_AllModels.csv  ({len(df_perturb)}行)")


# ============================================================
# 三、Bootstrap 重抽样验证（三模型各 N=100 次）
# ============================================================
print(f"\n[3/4] Bootstrap重抽样（N={N_BOOT}, 每次{BOOT_SAMPLE:,}像元）...")

actual_boot = min(BOOT_SAMPLE, n_px)
if actual_boot < BOOT_SAMPLE:
    print(f"  总像元数({n_px:,}) < 目标抽样数({BOOT_SAMPLE:,})，使用全量")

boot_results = {model: [] for model in MODEL_NAMES}

for model in MODEL_NAMES:
    print(f"\n  --- {model} ---")
    bks = breaks_dict[model]
    for i in range(N_BOOT):
        boot_idx = np.random.choice(n_px, actual_boot, replace=True)
        X_boot   = X_flat[boot_idx]

        s_boot       = compute_score(model, X_boot, w_combo)
        labels_new   = jenks_label(s_boot, bks)
        labels_orig  = level_flat[boot_idx]

        try:
            kappa = cohen_kappa_score(labels_orig, labels_new, labels=[1, 2, 3, 4, 5])
        except Exception:
            kappa = np.nan
        boot_results[model].append(float(kappa))

        if (i + 1) % 25 == 0:
            valid_k  = [k for k in boot_results[model] if not np.isnan(k)]
            cur_mean = np.mean(valid_k) if valid_k else np.nan
            print(f"    进度: {i+1}/{N_BOOT}  当前κ均值={cur_mean:.4f}")

# 汇总 Bootstrap 统计
boot_stats = {}
for model in MODEL_NAMES:
    kappas = [k for k in boot_results[model] if not np.isnan(k)]
    boot_stats[model] = {
        'mean': float(np.mean(kappas)),
        'std':  float(np.std(kappas)),
        'min':  float(np.min(kappas)),
        'pass': float(np.mean(kappas)) >= KAPPA_GOOD,
    }

print(f"\n  Bootstrap重抽样汇总（阈值 κ ≥ {KAPPA_GOOD}）:")
print(f"  {'模型':<10} {'κ均值':>10} {'κ标准差':>10} {'κ最小值':>10} {'结论':>8}")
print("  " + "-" * 52)
for model in MODEL_NAMES:
    st = boot_stats[model]
    flag = "通过" if st['pass'] else "未通过"
    print(f"  {model:<10} {st['mean']:>10.4f} {st['std']:>10.4f} "
          f"{st['min']:>10.4f} {flag:>8}")

# 保存 Bootstrap CSV
rows_b = {
    'Bootstrap次序': list(range(1, N_BOOT + 1)),
    'WLC_kappa':    boot_results['WLC'],
    'TOPSIS_kappa': boot_results['TOPSIS'],
    'VSC_kappa':    boot_results['VSC'],
}
df_boot = pd.DataFrame(rows_b)
summary_boot = pd.DataFrame([
    {'Bootstrap次序': '汇总_均值',
     'WLC_kappa':    boot_stats['WLC']['mean'],
     'TOPSIS_kappa': boot_stats['TOPSIS']['mean'],
     'VSC_kappa':    boot_stats['VSC']['mean']},
    {'Bootstrap次序': '汇总_标准差',
     'WLC_kappa':    boot_stats['WLC']['std'],
     'TOPSIS_kappa': boot_stats['TOPSIS']['std'],
     'VSC_kappa':    boot_stats['VSC']['std']},
    {'Bootstrap次序': '汇总_最小值',
     'WLC_kappa':    boot_stats['WLC']['min'],
     'TOPSIS_kappa': boot_stats['TOPSIS']['min'],
     'VSC_kappa':    boot_stats['VSC']['min']},
])
df_boot = pd.concat([df_boot, summary_boot], ignore_index=True)
out_b = os.path.join(OUTPUT_DIR, 'Bootstrap_AllModels.csv')
df_boot.to_csv(out_b, index=False, encoding='utf-8-sig')
print(f"\n  Bootstrap_AllModels.csv  ({len(df_boot)}行)")


# ============================================================
# 四、可视化：三模型稳健性对比柱状图
# ============================================================
print("\n[4/4] 生成三模型稳健性对比图...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('WLC / TOPSIS / VSC 三模型稳健性对比验证\n北京市洪涝风险评估（2012-2024）',
             fontsize=14, fontweight='bold', y=1.01)

x      = np.arange(len(MODEL_NAMES))
width  = 0.5
colors = [MODEL_COLORS[m] for m in MODEL_NAMES]

# ── 上图：权重扰动 Spearman ρ
rho_means = [perturb_stats[m]['mean'] for m in MODEL_NAMES]
rho_stds  = [perturb_stats[m]['std']  for m in MODEL_NAMES]

bars1 = ax1.bar(x, rho_means, width, yerr=rho_stds, color=colors,
                capsize=6, alpha=0.85, edgecolor='white', linewidth=1.2,
                error_kw={'elinewidth': 2, 'ecolor': '#333333', 'capthick': 2})
ax1.axhline(RHO_GOOD, color='#FF6B35', linewidth=2, linestyle='--',
            label=f'稳健阈值 ρ={RHO_GOOD}')
ax1.set_xticks(x)
ax1.set_xticklabels(MODEL_NAMES, fontsize=13)
ax1.set_ylabel('Spearman 秩相关系数 ρ', fontsize=12)
ax1.set_title(f'A. 权重扰动测试（N={N_PERTURB}次, ±{PERTURB_PCT*100:.0f}%扰动）',
              fontsize=12, fontweight='bold')
ax1.set_ylim(max(0, min(rho_means) - 0.05), 1.02)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# 在柱顶标注均值和通过/未通过
for i, (model, bar) in enumerate(zip(MODEL_NAMES, bars1)):
    st   = perturb_stats[model]
    flag = "通过" if st['pass'] else "未通过"
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + rho_stds[i] + 0.003,
             f"ρ={st['mean']:.4f}\n{flag}",
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             color='#1A1A1A')

# ── 下图：Bootstrap Cohen's Kappa
kappa_means = [boot_stats[m]['mean'] for m in MODEL_NAMES]
kappa_stds  = [boot_stats[m]['std']  for m in MODEL_NAMES]

bars2 = ax2.bar(x, kappa_means, width, yerr=kappa_stds, color=colors,
                capsize=6, alpha=0.85, edgecolor='white', linewidth=1.2,
                error_kw={'elinewidth': 2, 'ecolor': '#333333', 'capthick': 2})
ax2.axhline(KAPPA_GOOD, color='#FF6B35', linewidth=2, linestyle='--',
            label=f'稳健阈值 κ={KAPPA_GOOD}')
ax2.set_xticks(x)
ax2.set_xticklabels(MODEL_NAMES, fontsize=13)
ax2.set_ylabel("Cohen's Kappa 系数 κ", fontsize=12)
ax2.set_title(f'B. Bootstrap重抽样验证（N={N_BOOT}次, {actual_boot:,}像元/次）',
              fontsize=12, fontweight='bold')
ax2.set_ylim(max(0, min(kappa_means) - 0.1), 1.05)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

for i, (model, bar) in enumerate(zip(MODEL_NAMES, bars2)):
    st   = boot_stats[model]
    flag = "通过" if st['pass'] else "未通过"
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + kappa_stds[i] + 0.005,
             f"κ={st['mean']:.4f}\n{flag}",
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             color='#1A1A1A')

# 图例色块
legend_patches = [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in MODEL_NAMES]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out_fig = os.path.join(VIS_DIR, 'Robustness_AllModels_Report.png')
fig.savefig(out_fig, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Robustness_AllModels_Report.png")


# ============================================================
# 五、控制台汇总打印
# ============================================================
print("\n" + "=" * 70)
print("三模型稳健性验证汇总报告")
print(f"研究区：北京市  研究期：2012-2024  随机种子：2024")
print("=" * 70)
print(f"\n{'模型':<10} {'ρ均值':>10} {'ρ通过':>8}  {'κ均值':>10} {'κ通过':>8}")
print("-" * 54)
for model in MODEL_NAMES:
    ps = perturb_stats[model]
    bs = boot_stats[model]
    rho_flag   = "通过" if ps['pass'] else "未通过"
    kappa_flag = "通过" if bs['pass'] else "未通过"
    print(f"{model:<10} {ps['mean']:>10.6f} {rho_flag:>8}  "
          f"{bs['mean']:>10.4f} {kappa_flag:>8}")

print("\n稳健阈值：权重扰动 ρ ≥ 0.95  |  Bootstrap κ ≥ 0.80")

all_pass = all(perturb_stats[m]['pass'] and boot_stats[m]['pass']
               for m in MODEL_NAMES)
print(f"\n总体结论: {'三模型均通过稳健性验证，结果可信' if all_pass else '部分模型未达稳健阈值，结果解释需谨慎'}")

print("\n输出文件:")
print(f"  {out_p}")
print(f"  {out_b}")
print(f"  {out_fig}")
print("=" * 70)
