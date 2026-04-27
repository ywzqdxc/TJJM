"""
Step 5: 统计建模主程序（大赛版 v2）
=====================================
输入: Step4/All_Years_Dataset.csv

输出:
  Step5/model_comparison.csv         三模型 AUC 对比
  Step5/logistic_coefficients.csv    Logistic 回归系数 + OR + CI
  Step5/feature_importance.csv       随机森林特征重要性
  Step5/robustness_check.csv         稳健性检验（留一年交叉验证）
  Step5/statistical_tests.csv        VIF + 相关性 + HL 检验汇总
  Step5/model_results.png            ROC 曲线 + 特征重要性图

模型：
  M1: 多元 Logistic 回归（基准，7 个核心地形-水文特征）
  M2: 空间增强 Logistic 回归（主模型，含区位虚拟变量）
  M3: 随机森林（对比，提供特征重要性）

修复点（来自 step5_fix_and_rerun.py）：
  - 特征集精简为 7 个（rainfall_mm, RE_mm, HAND_m, TWI, slope_deg, dem_m, theta_s）
    去掉 ks_mmh / psi_cm / SM_prev / soil_deficit，这些在 VIF 检验中与其他特征高度共线
  - 随机森林收紧参数（max_depth=6, min_samples_leaf=5, min_samples_split=10）防止过拟合
  - HL 检验分组数改为 g=5（样本量较小时比 g=10 更稳定）
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from scipy.stats import ks_2samp
from numpy.linalg import pinv
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

OUTPUT_DIR = r'.\Step5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== [1] 读取数据 ====================
print("=" * 55)
print("[1] 读取数据...")
df = pd.read_csv(r'.\Step4\All_Years_Dataset.csv')
print(f"  总样本: {len(df)}  正:{(df['label']==1).sum()}  负:{(df['label']==0).sum()}")

# 核心特征（7个，VIF筛选后，去掉坐标/高共线列）
FEATURES = ['rainfall_mm', 'RE_mm', 'HAND_m', 'TWI', 'slope_deg', 'dem_m', 'theta_s']
TARGET   = 'label'
print(f"  建模特征({len(FEATURES)}个): {FEATURES}")

# ==================== [2] 时序划分 ====================
print("\n[2] 时序划分（严格按年份，禁止随机 shuffle）...")
train = df[df['year'].between(2018, 2022)].copy()
val   = df[df['year'] == 2023].copy()
test  = df[df['year'] == 2024].copy()

for name, d in [('训练集(2018-2022)', train), ('验证集(2023)', val), ('测试集(2024)', test)]:
    n = len(d); p = d['label'].sum()
    print(f"  {name}: {n}条（正{p}/负{n-p}）")

X_train = train[FEATURES].fillna(0).values
y_train = train[TARGET].values
X_val   = val[FEATURES].fillna(0).values
y_val   = val[TARGET].values
X_test  = test[FEATURES].fillna(0).values
y_test  = test[TARGET].values

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ==================== [3] VIF 多重共线性检验 ====================
print("\n[3] VIF 多重共线性检验...")


def calc_vif(X, names):
    res = []
    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_r = np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
        try:
            b      = pinv(X_r.T @ X_r) @ X_r.T @ y_i
            ss_res = np.sum((y_i - X_r @ b) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif    = 1 / (1 - r2) if r2 < 1 else 999
        except Exception:
            vif = 999
        res.append({'特征': names[i], 'VIF': round(vif, 2)})
    return pd.DataFrame(res).sort_values('VIF', ascending=False)


vif_df   = calc_vif(X_train_s, FEATURES)
high_vif = vif_df[vif_df['VIF'] > 10]['特征'].tolist()
print(vif_df.to_string(index=False))
print(f"  最大 VIF={vif_df['VIF'].max():.2f}  "
      f"{'✅ 全部<10' if not high_vif else f'⚠️ 高共线性: {high_vif}'}")

# ==================== [4] 点双列相关分析 ====================
print("\n[4] 点双列相关分析（特征与标签）...")
corr_res = []
for feat in FEATURES:
    r, p = stats.pointbiserialr(train[feat].fillna(0), y_train)
    corr_res.append({
        '特征': feat, '相关系数r': round(r, 4), 'P值': round(p, 4),
        '显著性': '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    })
corr_df = pd.DataFrame(corr_res).sort_values('相关系数r', key=abs, ascending=False)
print(corr_df.to_string(index=False))

# ==================== [5] 模型1：Logistic 基准 ====================
print("\n[5] 模型1：多元 Logistic 回归（基准）...")
lr1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr1.fit(X_train_s, y_train)

p_val_lr1  = lr1.predict_proba(X_val_s)[:, 1]
p_test_lr1 = lr1.predict_proba(X_test_s)[:, 1]
auc_val_1  = roc_auc_score(y_val,  p_val_lr1)  if y_val.sum()  > 0 else np.nan
auc_test_1 = roc_auc_score(y_test, p_test_lr1) if y_test.sum() > 0 else np.nan
print(f"  验证集 AUC={auc_val_1:.4f}  测试集 AUC={auc_test_1:.4f}")

# OR 值 + 95% CI
n_tr   = len(X_train_s)
coef   = lr1.coef_[0]
se     = np.sqrt(np.diag(pinv(X_train_s.T @ X_train_s + np.eye(len(FEATURES)) * 1e-6))) / np.sqrt(n_tr)
z_vals = coef / (se + 1e-10)
p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
coef_df = pd.DataFrame({
    '特征':    FEATURES,
    '回归系数': coef.round(4),
    'OR值':    np.exp(coef).round(4),
    '95%CI下': np.exp(coef - 1.96 * se).round(4),
    '95%CI上': np.exp(coef + 1.96 * se).round(4),
    'Z值':     z_vals.round(3),
    'P值':     p_vals.round(4),
    '显著性':  ['***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
               for p in p_vals]
}).sort_values('OR值', ascending=False)
print("\n  回归系数表:")
print(coef_df.to_string(index=False))
coef_df.to_csv(os.path.join(OUTPUT_DIR, 'logistic_coefficients.csv'),
               index=False, encoding='utf-8-sig')

# Hosmer-Lemeshow 检验（g=5，样本量小时比 g=10 更稳定）
def hosmer_lemeshow(y_true, y_prob, g=5):
    dhl = pd.DataFrame({'y': y_true, 'p': y_prob})
    dhl['dec'] = pd.qcut(dhl['p'], q=g, duplicates='drop')
    grp  = dhl.groupby('dec', observed=True)
    obs1 = grp['y'].sum()
    n    = grp['y'].count()
    exp1 = grp['p'].mean() * n
    obs0, exp0 = n - obs1, n - exp1
    hl = ((obs1 - exp1) ** 2 / exp1.clip(0.5) +
          (obs0 - exp0) ** 2 / exp0.clip(0.5)).sum()
    return round(hl, 3), round(1 - stats.chi2.cdf(hl, df=max(g - 2, 1)), 4)


p_train_lr1    = lr1.predict_proba(X_train_s)[:, 1]
hl_stat, hl_p  = hosmer_lemeshow(y_train, p_train_lr1, g=5)
print(f"\n  Hosmer-Lemeshow(g=5): χ²={hl_stat}, P={hl_p}  "
      f"({'✅ 拟合良好' if hl_p > 0.05 else '⚠️ 样本量大时 HL 易拒绝，参考 AUC 更可靠'})")

# ==================== [6] 模型2：空间增强 Logistic ====================
print("\n[6] 模型2：空间增强 Logistic（主模型）...")
for d in [train, val, test]:
    d['lat_bin'] = pd.cut(d['latitude'],  bins=5, labels=False)
    d['lon_bin'] = pd.cut(d['longitude'], bins=5, labels=False)

feat2 = FEATURES + ['lat_bin', 'lon_bin']
sc2   = StandardScaler()
X_tr2 = sc2.fit_transform(train[feat2].fillna(0))
X_va2 = sc2.transform(val[feat2].fillna(0))
X_te2 = sc2.transform(test[feat2].fillna(0))

lr2 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr2.fit(X_tr2, y_train)
p_val_lr2  = lr2.predict_proba(X_va2)[:, 1]
p_test_lr2 = lr2.predict_proba(X_te2)[:, 1]
auc_val_2  = roc_auc_score(y_val,  p_val_lr2)  if y_val.sum()  > 0 else np.nan
auc_test_2 = roc_auc_score(y_test, p_test_lr2) if y_test.sum() > 0 else np.nan
print(f"  验证集 AUC={auc_val_2:.4f}  测试集 AUC={auc_test_2:.4f}")

# ==================== [7] 模型3：随机森林（修复过拟合版） ====================
print("\n[7] 模型3：随机森林（修复过拟合，max_depth=6）...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=6,
    min_samples_leaf=5, min_samples_split=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

p_val_rf  = rf.predict_proba(X_val)[:, 1]
p_test_rf = rf.predict_proba(X_test)[:, 1]
auc_val_3  = roc_auc_score(y_val,  p_val_rf)  if y_val.sum()  > 0 else np.nan
auc_test_3 = roc_auc_score(y_test, p_test_rf) if y_test.sum() > 0 else np.nan
print(f"  验证集 AUC={auc_val_3:.4f}  测试集 AUC={auc_test_3:.4f}")

fi_df = pd.DataFrame({
    '特征': FEATURES,
    '重要性': rf.feature_importances_.round(4)
}).sort_values('重要性', ascending=False)
print("\n  特征重要性排序:")
print(fi_df.to_string(index=False))
fi_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'),
             index=False, encoding='utf-8-sig')

# ==================== [8] KS 统计量 ====================
print("\n[8] KS 统计量检验...")
for name, prob in [('Logistic基准', p_test_lr1),
                   ('空间Logistic', p_test_lr2),
                   ('随机森林',     p_test_rf)]:
    if y_test.sum() > 0 and (y_test == 0).sum() > 0:
        ks, ksp = ks_2samp(prob[y_test == 1], prob[y_test == 0])
        print(f"  {name}: KS={ks:.4f}  P={ksp:.4f}  {'✅' if ks > 0.3 else '⚠️'}")
    else:
        print(f"  {name}: 测试集只有单类，KS 无法计算")

# ==================== [9] 稳健性检验（留一年） ====================
print("\n[9] 稳健性检验（留一年交叉验证）...")
rob = []
for yr in sorted(df['year'].unique()):
    tr_ = df[df['year'] != yr]
    te_ = df[df['year'] == yr]
    if te_['label'].nunique() < 2:
        continue
    sc_ = StandardScaler()
    Xtr = sc_.fit_transform(tr_[FEATURES].fillna(0))
    Xte = sc_.transform(te_[FEATURES].fillna(0))
    m   = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    m.fit(Xtr, tr_['label'])
    auc = roc_auc_score(te_['label'], m.predict_proba(Xte)[:, 1])
    rob.append({'测试年份': yr, 'AUC': round(auc, 4)})

rob_df   = pd.DataFrame(rob)
rob_mean = rob_df['AUC'].mean()
rob_std  = rob_df['AUC'].std()
print(rob_df.to_string(index=False))
print(f"  均值={rob_mean:.4f}  标准差={rob_std:.4f}  "
      f"{'✅ 稳健（std<0.08）' if rob_std < 0.08 else '⚠️ 年份间差异较大'}")
rob_df.to_csv(os.path.join(OUTPUT_DIR, 'robustness_check.csv'),
              index=False, encoding='utf-8-sig')

# ==================== [10] 三模型对比表 ====================
comp_df = pd.DataFrame([
    {'模型': 'Logistic基准', 'AUC验证': round(auc_val_1, 4),
     'AUC测试': round(auc_test_1, 4), 'HL_P值': hl_p,
     '特点': '7特征，无空间效应（基准）'},
    {'模型': '空间Logistic', 'AUC验证': round(auc_val_2, 4),
     'AUC测试': round(auc_test_2, 4), 'HL_P值': '—',
     '特点': '7特征+区位虚拟变量（主模型）'},
    {'模型': '随机森林',     'AUC验证': round(auc_val_3, 4),
     'AUC测试': round(auc_test_3, 4), 'HL_P值': '—',
     '特点': '7特征，提供重要性排序（max_depth=6）'},
])
print(f"\n{'='*55}")
print("三模型对比:")
print(comp_df.to_string(index=False))
comp_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'),
               index=False, encoding='utf-8-sig')

# 统计检验汇总
stat_df = pd.DataFrame([
    {'检验类型': 'VIF多重共线性', '结果': f'高VIF特征:{high_vif if high_vif else "无"}',
     '结论': '✅ 无多重共线性' if not high_vif else '⚠️ 存在共线性'},
    {'检验类型': 'Hosmer-Lemeshow拟合优度', '结果': f'χ²={hl_stat}, P={hl_p}',
     '结论': '✅ 拟合良好' if hl_p > 0.05 else '⚠️ 参考AUC'},
    {'检验类型': '稳健性检验', '结果': f'AUC={rob_mean:.4f}±{rob_std:.4f}',
     '结论': '✅ 模型稳健' if rob_std < 0.08 else '⚠️ 年份间差异较大'},
])
stat_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_tests.csv'),
               index=False, encoding='utf-8-sig')

# ==================== [11] 绘图 ====================
print("\n生成论文图表...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC 曲线
ax = axes[0]
if y_test.sum() > 0 and (y_test == 0).sum() > 0:
    for name, prob, auc in [
        ('Logistic基准', p_test_lr1, auc_test_1),
        ('空间Logistic', p_test_lr2, auc_test_2),
        ('随机森林',     p_test_rf,  auc_test_3),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('假正率 FPR', fontsize=12)
ax.set_ylabel('真正率 TPR', fontsize=12)
ax.set_title('三模型 ROC 曲线对比（测试集 2024 年）', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# 特征重要性
ax = axes[1]
fi_plot = fi_df.sort_values('重要性')
colors  = ['#D85A30' if v > fi_df['重要性'].median() else '#5B9BD5'
           for v in fi_plot['重要性']]
bars = ax.barh(fi_plot['特征'], fi_plot['重要性'],
               color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, fi_plot['重要性']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)
ax.axvline(fi_df['重要性'].median(), color='gray', linestyle='--',
           linewidth=1, alpha=0.7, label='中位数')
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title('致涝因子重要性排序（随机森林）', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_results.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✅ model_results.png 已保存")

print(f"\n{'='*55}")
print(f"Step5 完成！输出文件:")
print(f"  model_comparison.csv       三模型对比")
print(f"  logistic_coefficients.csv  回归系数+OR+CI")
print(f"  feature_importance.csv     特征重要性")
print(f"  robustness_check.csv       稳健性检验")
print(f"  statistical_tests.csv      统计检验汇总")
print(f"  model_results.png          ROC+重要性图")
