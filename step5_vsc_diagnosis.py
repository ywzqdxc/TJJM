"""
Step 5 (v5.0): VSC三维脆弱性驱动因子统计差异诊断
==================================================
核心分析：
  1. Kruskal-Wallis 非参数检验（10指标 × 5等级）
  2. 小提琴图（10合1，含H统计量和P值标注）
  3. 致脆类型识别（8种，仿魏盛宇等2025论文）
  4. 致脆类型面积占比饼图 + 各等级内构成堆叠柱状图

输入：
  Step3: Risk_Level_4D.tif, Risk_Score_4D.tif
         Exposure/Sensitivity/CopingCapacity_Score.npy
  Step2.5b归一化后的各指标TIF（10个）

输出：
  Step5_KW_Violin.png          小提琴图（10指标）
  Step5_VulnType_Pie.png       致脆类型分析
  VSC_Risk_Driver_Summary.csv  各等级统计汇总
  VSC_VulnType_Statistics.csv  致脆类型面积占比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats
import rasterio
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

# ============================================================
# 路径配置
# ============================================================
STATIC_DIR = r'./Step_New/Static'
RISK_DIR   = r'./Step_New/Risk_Map'
EXT_DIR    = r'./Step_New/External'
DYN_DIR    = r'./Step_New/Dynamic'
VIS_DIR    = r'./Step_New/Visualization/Step5_VSC'
STAT_DIR   = r'./Step_New/Step5_Statistics'
for d in [VIS_DIR, STAT_DIR]: os.makedirs(d, exist_ok=True)

LEVEL_NAMES  = ['极低脆弱', '低脆弱', '中脆弱', '高脆弱', '极高脆弱']
LEVEL_COLORS = ['#2ECC71', '#A8E063', '#F39C12', '#E74C3C', '#8B0000']

print("=" * 70)
print("Step 5 (v5.0): VSC三维脆弱性驱动因子诊断")
print("=" * 70)


# ============================================================
# 工具函数
# ============================================================
def load_npy(path):
    if not os.path.exists(path): return None
    return np.load(path).astype(np.float32)

def load_tif(path):
    if not os.path.exists(path): return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        if nd is not None: arr = np.where(arr == nd, np.nan, arr)
    return np.where(arr < -1e10, np.nan, arr)


# ============================================================
# 一、加载数据
# ============================================================
print("\n[1/4] 加载数据...")

nodata_mask = load_npy(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

# 脆弱性指数与等级
vuln_score  = load_tif(os.path.join(RISK_DIR, 'Risk_Score_4D.tif'))
risk_level  = load_tif(os.path.join(RISK_DIR, 'Risk_Level_4D.tif'))

# 三维分量
E_score = load_npy(os.path.join(RISK_DIR, 'Exposure_Score.npy'))
S_score = load_npy(os.path.join(RISK_DIR, 'Sensitivity_Score.npy'))
C_score = load_npy(os.path.join(RISK_DIR, 'CopingCapacity_Score.npy'))

# 10个原始归一化指标（用于驱动因子差异分析）
# 优先从Step3生成的归一化数据推断，若无则从原始TIF重加载
ind_arrays = {
    '降雨量R':  load_npy(os.path.join(DYN_DIR, 'Precipitation_Mean.npy')),
    '径流CR':   load_npy(os.path.join(DYN_DIR, 'CR_MultiYear_Mean.npy')),
    'TWI':      load_npy(os.path.join(STATIC_DIR, 'twi.npy')),
    'HAND(m)':  load_npy(os.path.join(STATIC_DIR, 'hand.npy')),
    '积水点WP': load_tif(os.path.join(EXT_DIR, 'waterlogging_point_density_30m.tif')),
    '人口PD':   load_tif(os.path.join(EXT_DIR, 'population_density_30m.tif')),
    '道路RD':   load_tif(os.path.join(EXT_DIR, 'road_density_30m.tif')),
    '避难SH':   load_tif(os.path.join(EXT_DIR, 'shelter_density_30m.tif')),
    '医院HO':   load_tif(os.path.join(EXT_DIR, 'hospital_density_30m.tif')),
    '消防FS':   load_tif(os.path.join(EXT_DIR, 'firestation_density_30m.tif')),
}

# 构建有效掩膜
valid_full = valid_mask & ~np.isnan(risk_level) & ~np.isnan(vuln_score)
print(f"  有效像元: {valid_full.sum():,}")

# 构建DataFrame
data_dict = {'综合脆弱性': vuln_score[valid_full],
             'Risk_Level': risk_level[valid_full].astype(int)}
for name, arr in ind_arrays.items():
    data_dict[name] = arr[valid_full] if arr is not None else np.full(valid_full.sum(), np.nan)
if E_score is not None: data_dict['暴露度E'] = E_score[valid_full]
if S_score is not None: data_dict['敏感性S'] = S_score[valid_full]
if C_score is not None: data_dict['应对不足C'] = C_score[valid_full]

df = pd.DataFrame(data_dict).dropna(subset=['综合脆弱性', 'Risk_Level'])
df = df[df['Risk_Level'].between(1, 5)]
print(f"  分析样本: {len(df):,} 像元")


# ============================================================
# 二、Kruskal-Wallis 非参数检验
# ============================================================
print("\n[2/4] Kruskal-Wallis 非参数检验...")

# 选择分析指标（10原始指标 + 3维度分量）
kw_cols = list(ind_arrays.keys()) + ['暴露度E', '敏感性S', '应对不足C']
kw_cols = [c for c in kw_cols if c in df.columns]

kw_results = {}
print(f"  {'指标':<14}{'H统计量':>10}{'P值':>12}{'显著性':>8}")
print("  " + "-"*46)
for col in kw_cols:
    groups = [df[df['Risk_Level']==lv][col].dropna().values for lv in range(1,6)]
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) < 2:
        kw_results[col] = {'H': np.nan, 'P': np.nan}
        continue
    h_stat, p_val = stats.kruskal(*groups)
    kw_results[col] = {'H': h_stat, 'P': p_val}
    sig = '***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else 'ns'))
    print(f"  {col:<14}{h_stat:>10.2f}{p_val:>12.2e}{sig:>8}")


# ============================================================
# 三、小提琴图（10原始指标 + 3维度分量）
# ============================================================
print("\n[3/4] 绘制小提琴图...")

# 分两批：图3a 10原始指标（2×5），图3b 3维度分量（1×3）
def violin_panel(fig, axes_flat, cols_to_plot, df, kw_results, title):
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    for idx, col in enumerate(cols_to_plot):
        ax = axes_flat[idx] if idx < len(axes_flat) else None
        if ax is None: break
        plot_data = [df[df['Risk_Level']==lv][col].dropna().values
                     for lv in range(1,6)]
        if all(len(g) > 5 for g in plot_data):
            parts = ax.violinplot(plot_data, positions=range(1,6),
                                   showmeans=False, showmedians=True, showextrema=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(LEVEL_COLORS[i]); pc.set_alpha(0.75)
                pc.set_edgecolor('black'); pc.set_linewidth(0.5)
            parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
            parts['cmaxes'].set_color('grey');   parts['cmins'].set_color('grey')
        res = kw_results.get(col, {'H': np.nan, 'P': np.nan})
        if not np.isnan(res['H']):
            sig_s = '***' if res['P']<0.001 else ('**' if res['P']<0.01 else '*')
            ax.text(0.5, 0.95, f"H={res['H']:.1f}\nP={res['P']:.2e} {sig_s}",
                    transform=ax.transAxes, fontsize=8, va='top', ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
        ax.set_xticks(range(1,6))
        ax.set_xticklabels(['极低','低','中','高','极高'], rotation=20, fontsize=7.5)
        ax.set_title(f'【{col}】', fontsize=10, fontweight='bold')
        ax.set_ylabel(col, fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if idx == 0:
            patches = [mpatches.Patch(facecolor=c, alpha=0.75, label=n)
                       for c, n in zip(LEVEL_COLORS, LEVEL_NAMES)]
            ax.legend(handles=patches, loc='upper right', fontsize=6,
                      framealpha=0.9, title='脆弱性等级', title_fontsize=7)

# 图5a：10原始指标
raw_cols = list(ind_arrays.keys())
fig5a, axes5a = plt.subplots(2, 5, figsize=(22, 10), facecolor='white')
violin_panel(fig5a, axes5a.flat, raw_cols, df, kw_results,
             'VSC三维内涝脆弱性——10原始指标分布差异诊断（北京市 2012-2024）')
plt.tight_layout(rect=[0,0,1,0.97])
out_5a = os.path.join(VIS_DIR, 'Step5_KW_Violin_Raw.png')
fig5a.savefig(out_5a, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Step5_KW_Violin_Raw.png")

# 图5b：3维度分量
dim_cols = [c for c in ['暴露度E', '敏感性S', '应对不足C'] if c in df.columns]
if dim_cols:
    fig5b, axes5b = plt.subplots(1, 3, figsize=(14, 6), facecolor='white')
    violin_panel(fig5b, axes5b.flat, dim_cols, df, kw_results,
                 'VSC三维分量（E/S/C）在不同脆弱性等级下的分布差异')
    plt.tight_layout(rect=[0,0,1,0.97])
    out_5b = os.path.join(VIS_DIR, 'Step5_KW_Violin_ESC.png')
    fig5b.savefig(out_5b, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Step5_KW_Violin_ESC.png")


# ============================================================
# 四、致脆类型识别（8种，基于E/S/C三维框架）
# ============================================================
print("\n[4/4] 致脆类型识别...")

if all(c in df.columns for c in ['暴露度E', '敏感性S', '应对不足C']):
    # 以各维度全局中位数为阈值
    E_thr = float(df['暴露度E'].median())
    S_thr = float(df['敏感性S'].median())
    C_thr = float(df['应对不足C'].median())
    print(f"  致脆阈值: E_中位={E_thr:.4f}  S_中位={S_thr:.4f}  C_中位={C_thr:.4f}")

    df['E_flag'] = (df['暴露度E']   > E_thr).astype(int)
    df['S_flag'] = (df['敏感性S']   > S_thr).astype(int)
    df['C_flag'] = (df['应对不足C'] > C_thr).astype(int)

    TYPE_ORDER = ['O（弱综合型）', 'E（暴露致脆）', 'S（敏感致脆）', 'C（应对不足）',
                  'ES（暴露-敏感）', 'EC（暴露-应对）', 'SC（敏感-应对）', 'ESC（强综合型）']
    TYPE_COLORS = ['#95A5A6', '#E74C3C', '#F39C12', '#3498DB',
                   '#8E44AD', '#E67E22', '#2ECC71', '#1ABC9C']

    def classify(row):
        e, s, c = row['E_flag'], row['S_flag'], row['C_flag']
        n = e + s + c
        if n == 0: return 'O（弱综合型）'
        if n == 1:
            if e: return 'E（暴露致脆）'
            if s: return 'S（敏感致脆）'
            return 'C（应对不足）'
        if n == 2:
            if not e: return 'SC（敏感-应对）'
            if not s: return 'EC（暴露-应对）'
            return 'ES（暴露-敏感）'
        return 'ESC（强综合型）'

    df['VulnType'] = df.apply(classify, axis=1)

    # 统计
    type_stats = df.groupby('VulnType').agg(
        像元数=('综合脆弱性','count'),
        脆弱性均值=('综合脆弱性','mean'),
        脆弱性Std=('综合脆弱性','std')
    ).reset_index()
    type_stats['面积占比%'] = type_stats['像元数'] / len(df) * 100
    type_stats = type_stats.sort_values('面积占比%', ascending=False)
    print(f"\n  致脆类型面积占比:")
    print(type_stats[['VulnType','像元数','面积占比%','脆弱性均值']].to_string(index=False,
          float_format='{:.4f}'.format))

    # 各等级内构成
    lv_type = pd.crosstab(df['Risk_Level'], df['VulnType'], normalize='index') * 100
    print(f"\n  各脆弱等级内致脆类型占比(%):")
    print(lv_type.round(1).to_string())

    type_stats.to_csv(os.path.join(STAT_DIR, 'VSC_VulnType_Statistics.csv'),
                      index=False, encoding='utf-8-sig')
    lv_type.round(2).to_csv(os.path.join(STAT_DIR, 'VSC_LevelType_Crosstab.csv'),
                              encoding='utf-8-sig')

    # ── 图：致脆类型饼图 + 堆叠柱状图 ──
    fig6, axes6 = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig6.suptitle('北京市内涝脆弱性致脆类型分析（VSC三维框架，2012-2024）',
                  fontsize=14, fontweight='bold')

    # 饼图
    counts = [df['VulnType'].value_counts().get(t, 0) for t in TYPE_ORDER]
    valid_types = [(t, c, col) for t, c, col in zip(TYPE_ORDER, counts, TYPE_COLORS)
                   if c > 0]
    if valid_types:
        t_labs, t_cnts, t_cols = zip(*valid_types)
        axes6[0].pie(t_cnts, labels=None, colors=t_cols,
                      autopct='%1.1f%%', startangle=90,
                      textprops={'fontsize': 9}, pctdistance=0.78,
                      wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        patches6 = [mpatches.Patch(color=c, label=t)
                    for t, c in zip(t_labs, t_cols)]
        axes6[0].legend(handles=patches6, loc='lower left', fontsize=7.5,
                         framealpha=0.9, bbox_to_anchor=(-0.05, -0.1))
        axes6[0].set_title('全域致脆类型面积占比', fontsize=12, fontweight='bold')

    # 堆叠柱状图
    lv_arr = np.zeros((5, len(TYPE_ORDER)))
    for ki, t in enumerate(TYPE_ORDER):
        for lv in range(1, 6):
            mask_lv = df['Risk_Level'] == lv
            total_lv = mask_lv.sum()
            n = (df.loc[mask_lv, 'VulnType'] == t).sum()
            lv_arr[lv-1, ki] = (n / total_lv * 100) if total_lv > 0 else 0

    bottom = np.zeros(5)
    for ki, (t, col) in enumerate(zip(TYPE_ORDER, TYPE_COLORS)):
        vals = lv_arr[:, ki]
        if vals.max() > 0.5:
            axes6[1].bar(range(1, 6), vals, bottom=bottom,
                          color=col, alpha=0.85, label=t, edgecolor='white')
        bottom += vals

    axes6[1].set_xticks(range(1, 6))
    axes6[1].set_xticklabels(LEVEL_NAMES, rotation=20, fontsize=9, ha='right')
    axes6[1].set_ylabel('类型占比 (%)', fontsize=11)
    axes6[1].set_ylim(0, 105)
    axes6[1].set_title('各脆弱等级内致脆类型构成', fontsize=12, fontweight='bold')
    axes6[1].legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
    axes6[1].grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_6 = os.path.join(VIS_DIR, 'Step5_VulnType_Analysis.png')
    fig6.savefig(out_6, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Step5_VulnType_Analysis.png")

else:
    print("  ⚠️  E/S/C分量未找到，跳过致脆类型识别")


# ============================================================
# 五、统计汇总表
# ============================================================
print("\n  生成统计汇总表...")

summary_cols = list(ind_arrays.keys()) + ['暴露度E', '敏感性S', '应对不足C', '综合脆弱性']
summary_cols = [c for c in summary_cols if c in df.columns]

summary_list = []
for lv in range(1, 6):
    mask_lv = df['Risk_Level'] == lv
    row = {'脆弱性等级': LEVEL_NAMES[lv-1], '像元数': int(mask_lv.sum())}
    for col in summary_cols:
        vals = df.loc[mask_lv, col].dropna().values
        row[f'{col}_mean'] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        row[f'{col}_std']  = float(np.std(vals))  if len(vals) > 0 else np.nan
    summary_list.append(row)

df_summary = pd.DataFrame(summary_list)
out_csv = os.path.join(STAT_DIR, 'VSC_Risk_Driver_Summary.csv')
df_summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"  ✅ VSC_Risk_Driver_Summary.csv")

# KW结果汇总
kw_df = pd.DataFrame([
    {'指标': k, 'H统计量': v['H'], 'P值': v['P'],
     '显著性': '***' if v['P']<0.001 else ('**' if v['P']<0.01
               else ('*' if v['P']<0.05 else 'ns')) if not np.isnan(v['P']) else 'NaN'}
    for k, v in kw_results.items()
])
kw_df.to_csv(os.path.join(STAT_DIR, 'VSC_KW_Results.csv'),
             index=False, encoding='utf-8-sig')
print(f"  ✅ VSC_KW_Results.csv")

print(f"\n✅ Step 5 (v5.0) 完成！")
print(f"  📊 小提琴图  → {VIS_DIR}/Step5_KW_Violin_Raw.png")
print(f"  📊 小提琴图  → {VIS_DIR}/Step5_KW_Violin_ESC.png")
print(f"  📊 致脆类型  → {VIS_DIR}/Step5_VulnType_Analysis.png")
print(f"  📋 统计表    → {STAT_DIR}/")
print("=" * 70)