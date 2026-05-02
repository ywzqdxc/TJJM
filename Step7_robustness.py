"""
Step 7: 稳健性验证
===================
A. 权重扰动测试（Weight Perturbation）
   对组合权重施加 N=100 次 ±10% 随机扰动，
   计算每次TOPSIS得分与原始得分的Spearman相关系数。
   结论：均值>0.95 → 权重选择稳健

B. Bootstrap重抽样稳定性验证
   从有效像元有放回抽取 BOOT_SAMPLE 个样本，重复 N_BOOT=100 次，
   计算每次脆弱性等级与原始等级的Cohen's Kappa系数。
   结论：均值>0.80 → 评估结果稳健

输出：
  Robustness/Weight_Perturbation_Results.csv
  Robustness/Bootstrap_Results.csv
  Robustness/Robustness_Report.png
  Robustness/Robustness_Summary.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
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
OUTPUT_DIR = r'./Step_New/Robustness'
VIS_DIR    = r'./Step_New/Visualization/Step7_Robustness'
for d in [OUTPUT_DIR, VIS_DIR]: os.makedirs(d, exist_ok=True)

# 超参数
N_PERTURB   = 100       # 权重扰动次数
PERTURB_PCT = 0.10      # 扰动幅度 ±10%
N_BOOT      = 100       # Bootstrap次数
BOOT_SAMPLE = 500_000   # Bootstrap每次抽样像元数（50万，约全量的2.3%）
KAPPA_GOOD  = 0.80      # Kappa稳健阈值
RHO_GOOD    = 0.95      # Spearman ρ稳健阈值

np.random.seed(2024)   # 固定随机种子，保证可重复

print("=" * 70)
print("Step 7: 稳健性验证")
print(f"A. 权重扰动测试: N={N_PERTURB}次, 扰动幅度±{PERTURB_PCT*100:.0f}%")
print(f"B. Bootstrap重抽样: N={N_BOOT}次, 每次{BOOT_SAMPLE:,}像元")
print("=" * 70)


# ============================================================
# 工具函数
# ============================================================

def load_npy(path):
    if not os.path.exists(path): return None
    return np.load(path).astype(np.float32)

def load_tif(path, h=None, w=None, dst_crs=None, dst_transform=None):
    if not os.path.exists(path): return None
    with rasterio.open(path) as src:
        if h and src.height != h:
            arr = np.full((h, w), -9999.0, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=arr,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, dst_nodata=-9999.0)
        else:
            arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None: arr = np.where(arr == nd, np.nan, arr)
    return np.where(arr < -1e10, np.nan, arr)

def minmax_norm(arr, valid_mask, direction='positive'):
    result = np.full_like(arr, np.nan, dtype=np.float32)
    vals   = arr[valid_mask & np.isfinite(arr)]
    if vals.size < 10: return result
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-10:
        result[valid_mask & np.isfinite(arr)] = 0.5; return result
    normed = np.clip((arr - mn) / (mx - mn), 0.0, 1.0)
    if direction == 'negative': normed = 1.0 - normed
    result[valid_mask & np.isfinite(arr)] = normed[valid_mask & np.isfinite(arr)]
    return result

def topsis(X, w):
    Xw    = X * w[np.newaxis, :]
    Z_pos = Xw.max(axis=0); Z_neg = Xw.min(axis=0)
    D_pos = np.sqrt(((Xw - Z_pos)**2).sum(axis=1))
    D_neg = np.sqrt(((Xw - Z_neg)**2).sum(axis=1))
    return (D_neg / (D_pos + D_neg + 1e-10)).astype(np.float32)

def jenks_label(scores, breaks):
    """用已有断点对得分分级（不重算断点）"""
    inner = breaks[1:-1]
    labels = np.searchsorted(inner, scores, side='left') + 1
    return np.clip(labels, 1, 5).astype(np.int8)


# ============================================================
# 一、加载基准数据
# ============================================================
print("\n[1/4] 加载基准数据...")

with rasterio.open(DEM_PATH) as ref:
    h, w         = ref.height, ref.width
    dst_crs       = ref.crs
    dst_transform = ref.transform

nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

# 加载权重（从CSV）
weights_csv = os.path.join(RISK_DIR, 'VSC_Weights.csv')
if not os.path.exists(weights_csv):
    raise FileNotFoundError(f"VSC_Weights.csv not found. Run step3_v6.py first.")
df_w     = pd.read_csv(weights_csv)
w_combo  = df_w['组合'].values.astype(np.float64)
m        = len(w_combo)   # 10个指标
print(f"  组合权重（{m}个指标）: {[f'{v:.4f}' for v in w_combo]}")

# 加载原始最优得分（TOPSIS）
s_optimal_grid = load_tif(os.path.join(RISK_DIR, 'Risk_Score_Optimal.tif'))
if s_optimal_grid is None:
    s_optimal_grid = load_tif(os.path.join(RISK_DIR, 'Vuln_TOPSIS.npy'.replace('.npy','.tif')))

# 加载原始等级
level_grid = load_tif(os.path.join(RISK_DIR, 'Risk_Level_Optimal.tif'))

# 加载Jenks断点（从Annual_Vuln_Stats或重算）
breaks_path = os.path.join(RISK_DIR, 'Jenks_Breaks.npy')
if os.path.exists(breaks_path):
    jenks_bks = np.load(breaks_path)
    print(f"  Jenks断点（从文件）: {[f'{v:.4f}' for v in jenks_bks]}")
else:
    # 从原始得分重算
    s_v = s_optimal_grid[valid_mask & np.isfinite(s_optimal_grid)]
    from scipy.stats import mstats
    jenks_bks = np.quantile(s_v, np.linspace(0, 1, 6))
    print(f"  Jenks断点（重算）: {[f'{v:.4f}' for v in jenks_bks]}")

# 重建归一化指标矩阵 X_flat
print(f"\n  重建指标矩阵 X_flat...")

rain = load_npy(os.path.join(DYN_DIR, 'Precipitation_Mean.npy'))
cr   = load_npy(os.path.join(DYN_DIR, 'CR_MultiYear_Mean.npy'))
twi  = load_npy(os.path.join(STATIC_DIR, 'twi.npy'))
hand = load_npy(os.path.join(STATIC_DIR, 'hand.npy'))
wp   = load_tif(os.path.join(EXT_DIR, 'waterlogging_point_density_30m.tif'))
pop  = load_tif(os.path.join(EXT_DIR, 'population_density_30m.tif'))
road = load_tif(os.path.join(EXT_DIR, 'road_density_30m.tif'))
sh   = load_tif(os.path.join(EXT_DIR, 'shelter_density_30m.tif'))
ho   = load_tif(os.path.join(EXT_DIR, 'hospital_density_30m.tif'))
fs   = load_tif(os.path.join(EXT_DIR, 'firestation_density_30m.tif'))

raw_arrs  = [rain, cr, twi, hand, wp, pop, road, sh, ho, fs]
dirs      = ['positive']*3 + ['negative'] + ['positive']*3 + ['negative']*3

norm_list = []
for arr, dire in zip(raw_arrs, dirs):
    if arr is None:
        norm_list.append(np.where(valid_mask, 0.0, np.nan).astype(np.float32))
    else:
        norm_list.append(minmax_norm(arr, valid_mask, dire))

all_valid_px = valid_mask.copy()
for arr in norm_list: all_valid_px &= np.isfinite(arr)

n_px   = int(all_valid_px.sum())
X_flat = np.column_stack([arr[all_valid_px] for arr in norm_list]).astype(np.float64)
X_flat = np.clip(X_flat, 0.0, 1.0)
print(f"  X_flat: {X_flat.shape}  完整有效像元={n_px:,}")

# 原始TOPSIS得分（从X_flat计算，保证与扰动基准一致）
s_base = topsis(X_flat, w_combo)

# 原始等级标签（从level_grid提取）
if level_grid is not None:
    level_flat = level_grid[all_valid_px].astype(np.int8)
    level_flat = np.clip(level_flat, 1, 5)
else:
    level_flat = jenks_label(s_base, jenks_bks)

print(f"  基准得分: 均值={s_base.mean():.4f}  Std={s_base.std():.4f}")
print(f"  等级分布: {dict(zip(*np.unique(level_flat, return_counts=True)))}")


# ============================================================
# 二、权重扰动测试
# ============================================================
print(f"\n[2/4] 权重扰动测试（N={N_PERTURB}, ±{PERTURB_PCT*100:.0f}%）...")

perturb_rhos = []

for i in range(N_PERTURB):
    # 生成扰动权重
    delta    = np.random.uniform(-PERTURB_PCT, PERTURB_PCT, size=m) * w_combo
    w_perturb = w_combo + delta
    w_perturb = np.clip(w_perturb, 0.0, None)       # 非负
    w_perturb = w_perturb / (w_perturb.sum() + 1e-12)  # 归一化

    # 计算扰动后TOPSIS得分
    s_perturb = topsis(X_flat, w_perturb)

    # Spearman相关
    rho, _ = spearmanr(s_base, s_perturb)
    perturb_rhos.append(float(rho))

    if (i + 1) % 25 == 0:
        print(f"  进度: {i+1}/{N_PERTURB}  当前ρ均值={np.mean(perturb_rhos):.6f}")

rho_mean = float(np.mean(perturb_rhos))
rho_std  = float(np.std(perturb_rhos))
rho_min  = float(np.min(perturb_rhos))
perturb_ok = rho_mean >= RHO_GOOD

print(f"\n  权重扰动测试结果:")
print(f"    Spearman ρ 均值  = {rho_mean:.6f}  {'✅ 稳健' if perturb_ok else '❌ 不稳健'}")
print(f"    Spearman ρ 标准差 = {rho_std:.6f}")
print(f"    Spearman ρ 最小值 = {rho_min:.6f}")
print(f"    稳健阈值: ρ ≥ {RHO_GOOD}")

df_perturb = pd.DataFrame({
    '扰动次序': range(1, N_PERTURB + 1),
    'Spearman_rho': perturb_rhos
})
df_perturb.loc[len(df_perturb)] = ['汇总_均值', rho_mean]
df_perturb.loc[len(df_perturb)] = ['汇总_标准差', rho_std]
df_perturb.loc[len(df_perturb)] = ['汇总_最小值', rho_min]
df_perturb.to_csv(os.path.join(OUTPUT_DIR, 'Weight_Perturbation_Results.csv'),
                   index=False, encoding='utf-8-sig')
print(f"  ✅ Weight_Perturbation_Results.csv")


# ============================================================
# 三、Bootstrap重抽样验证
# ============================================================
print(f"\n[3/4] Bootstrap重抽样（N={N_BOOT}, 每次{BOOT_SAMPLE:,}像元）...")

# 限制抽样数量不超过总像元数
actual_boot_sample = min(BOOT_SAMPLE, n_px)
if actual_boot_sample < BOOT_SAMPLE:
    print(f"  ⚠️  总像元数({n_px:,}) < 目标抽样数({BOOT_SAMPLE:,})，使用全量")

boot_kappas = []
idx_all     = np.arange(n_px)

for i in range(N_BOOT):
    # 有放回抽样
    boot_idx = np.random.choice(n_px, actual_boot_sample, replace=True)

    # 计算Bootstrap样本的TOPSIS得分
    X_boot    = X_flat[boot_idx]
    s_boot    = topsis(X_boot, w_combo)

    # 分级（使用原始Jenks断点，不重算）
    labels_boot_orig = level_flat[boot_idx]
    labels_boot_new  = jenks_label(s_boot, jenks_bks)

    # Cohen's Kappa
    try:
        kappa = cohen_kappa_score(labels_boot_orig, labels_boot_new,
                                   labels=[1, 2, 3, 4, 5])
    except Exception:
        kappa = np.nan

    boot_kappas.append(float(kappa))

    if (i + 1) % 25 == 0:
        valid_k = [k for k in boot_kappas if not np.isnan(k)]
        print(f"  进度: {i+1}/{N_BOOT}  当前Kappa均值={np.mean(valid_k):.4f}")

valid_kappas = [k for k in boot_kappas if not np.isnan(k)]
kappa_mean = float(np.mean(valid_kappas))
kappa_std  = float(np.std(valid_kappas))
kappa_min  = float(np.min(valid_kappas))
boot_ok    = kappa_mean >= KAPPA_GOOD

print(f"\n  Bootstrap重抽样结果:")
print(f"    Cohen's Kappa 均值  = {kappa_mean:.4f}  {'✅ 稳健' if boot_ok else '❌ 不稳健'}")
print(f"    Cohen's Kappa 标准差 = {kappa_std:.4f}")
print(f"    Cohen's Kappa 最小值 = {kappa_min:.4f}")
print(f"    稳健阈值: Kappa ≥ {KAPPA_GOOD}")

df_boot = pd.DataFrame({
    'Bootstrap次序': range(1, N_BOOT + 1),
    'Cohen_Kappa': boot_kappas
})
df_boot.loc[len(df_boot)] = ['汇总_均值', kappa_mean]
df_boot.loc[len(df_boot)] = ['汇总_标准差', kappa_std]
df_boot.loc[len(df_boot)] = ['汇总_最小值', kappa_min]
df_boot.to_csv(os.path.join(OUTPUT_DIR, 'Bootstrap_Results.csv'),
                index=False, encoding='utf-8-sig')
print(f"  ✅ Bootstrap_Results.csv")


# ============================================================
# 四、可视化与报告
# ============================================================
print("\n[4/4] 生成稳健性报告图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('VSC内涝脆弱性评估稳健性验证报告（北京市 2012-2024）',
              fontsize=14, fontweight='bold', y=1.02)

# ── 左图：权重扰动Spearman ρ分布
ax1 = axes[0]
ax1.hist(perturb_rhos, bins=30, color='#2C7BB6', alpha=0.8, edgecolor='white')

# 恢复图例中的具体数值
ax1.axvline(rho_mean, color='#D7191C', linewidth=2.5, linestyle='-',
            label=f'均值 ρ={rho_mean:.4f}')
ax1.axvline(RHO_GOOD, color='#1A9641', linewidth=2, linestyle='--',
            label=f'稳健阈值 {RHO_GOOD}')
ax1.fill_betweenx([0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 30],
                   rho_mean - rho_std, rho_mean + rho_std,
                   alpha=0.2, color='#2C7BB6',
                   label=f'±σ ({rho_std:.4f})')

ax1.set_xlabel('Spearman 秩相关系数 ρ', fontsize=12)
ax1.set_ylabel('频次', fontsize=12)
# 保持无 emoji 状态，防止出现乱码方框
ax1.set_title(f'A. 权重扰动测试（N={N_PERTURB}, ±{PERTURB_PCT*100:.0f}%扰动）\n'
               f'结论: {"稳健（均值>"+str(RHO_GOOD)+"）" if perturb_ok else "不稳健"}',
               fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)

# ── 右图：Bootstrap Kappa分布
ax2 = axes[1]
ax2.hist(valid_kappas, bins=30, color='#D7191C', alpha=0.8, edgecolor='white')
ax2.axvline(kappa_mean, color='#2C7BB6', linewidth=2.5, linestyle='-',
             label=f'均值 κ={kappa_mean:.4f}')
ax2.axvline(KAPPA_GOOD, color='#1A9641', linewidth=2, linestyle='--',
             label=f'稳健阈值 {KAPPA_GOOD}')
ax2.fill_betweenx([0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 30],
                   kappa_mean - kappa_std, kappa_mean + kappa_std,
                   alpha=0.2, color='#D7191C', label=f'±σ ({kappa_std:.4f})')
ax2.set_xlabel("Cohen's Kappa 系数", fontsize=12)
ax2.set_ylabel('频次', fontsize=12)
# 同理，去掉右图的 emoji 符号以防出现方框
ax2.set_title(f'B. Bootstrap重抽样（N={N_BOOT}, {actual_boot_sample:,}像元/次）\n'
               f'结论: {"稳健（均值>"+str(KAPPA_GOOD)+"）" if boot_ok else "不稳健"}',
               fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'Robustness_Report.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Robustness_Report.png")

# 文字报告
report_lines = [
    "=" * 60,
    "VSC内涝脆弱性评估稳健性验证报告",
    f"研究区：北京市  研究期：2012-2024  随机种子：2024",
    "=" * 60,
    "",
    "A. 权重扰动测试",
    f"   扰动次数: {N_PERTURB}次",
    f"   扰动幅度: ±{PERTURB_PCT*100:.0f}%（对每个权重分量独立扰动）",
    f"   Spearman ρ 均值: {rho_mean:.6f}",
    f"   Spearman ρ 标准差: {rho_std:.6f}",
    f"   Spearman ρ 最小值: {rho_min:.6f}",
    f"   稳健阈值: ρ ≥ {RHO_GOOD}",
    f"   结论: {'✅ 权重选择稳健' if perturb_ok else '❌ 权重选择不稳健，建议重新校验AHP判断矩阵'}",
    "",
    "B. Bootstrap重抽样稳定性验证",
    f"   抽样次数: {N_BOOT}次",
    f"   每次样本量: {actual_boot_sample:,}像元（有放回抽样）",
    f"   Cohen's Kappa 均值: {kappa_mean:.4f}",
    f"   Cohen's Kappa 标准差: {kappa_std:.4f}",
    f"   Cohen's Kappa 最小值: {kappa_min:.4f}",
    f"   稳健阈值: κ ≥ {KAPPA_GOOD}",
    f"   结论: {'✅ 评估结果稳健，等级分类具有良好重现性' if boot_ok else '❌ 评估结果不稳健，等级分类对样本较敏感'}",
    "",
    "综合结论:",
    f"  权重稳健性: {'✅ 通过' if perturb_ok else '❌ 未通过'}  (ρ={rho_mean:.4f})",
    f"  样本稳健性: {'✅ 通过' if boot_ok else '❌ 未通过'}  (κ={kappa_mean:.4f})",
    f"  总体评价: {'✅ 模型稳健，结果可信' if (perturb_ok and boot_ok) else '⚠️ 部分指标未达稳健阈值，结果解释需谨慎'}",
    "=" * 60,
]
report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(os.path.join(OUTPUT_DIR, 'Robustness_Summary.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\n  ✅ Robustness_Summary.txt")

print(f"\n✅ Step 7 完成！")
print(f"  {OUTPUT_DIR}/")
print(f"    Weight_Perturbation_Results.csv")
print(f"    Bootstrap_Results.csv")
print(f"    Robustness_Summary.txt")
print(f"  {VIS_DIR}/")
print(f"    Robustness_Report.png")