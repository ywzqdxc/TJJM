"""
Step 5 (v6.0): VSC三维风险驱动因子统计差异诊断
==================================================
v6.0核心变化（相对v5.0）：
  1. 读取 Risk_Score_Optimal.tif / Risk_Level_Optimal.tif
     （fallback到 Risk_Score_4D.tif / Risk_Level_4D.tif）
  2. ★ 小提琴图抽样优化：
     - Kruskal-Wallis检验：全量数百万像元（保证学术严谨性）
     - violin_panel绘图：每等级最多抽样 MAX_SAMPLES=20000 个像元
     - 两者完全隔离，统计结论不受绘图数据影响
  3. 致脆类型识别保持不变

性能说明：
  全量KW检验（~4000万像元）：~30秒（不变）
  小提琴图绘制（每等级20000样本）：~10秒（原~15分钟 → 大幅提升）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ★ 小提琴图抽样上限（每个风险等级的最大绘图样本数）
MAX_SAMPLES = 20000

LEVEL_NAMES  = ['极低风险', '低风险', '中风险', '高风险', '极高风险']
LEVEL_COLORS = ['#2ECC71', '#A8E063', '#F39C12', '#E74C3C', '#8B0000']

print("=" * 70)
print("Step 5 (v6.0): VSC三维风险驱动因子诊断")
print(f"★ 小提琴图抽样优化: MAX_SAMPLES={MAX_SAMPLES}/等级")
print("  KW检验: 全量数据 | 绘图: 抽样渲染（统计结论不变）")
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

def load_best(primary_path, fallback_path, label):
    arr = load_tif(primary_path)
    if arr is None:
        arr = load_tif(fallback_path)
        if arr is not None:
            print(f"  ⚠️  {label}: 使用fallback {os.path.basename(fallback_path)}")
        else:
            print(f"  ❌ {label}: 主路径和fallback均不存在！")
    else:
        print(f"  ✅ {label}: {os.path.basename(primary_path)}")
    return arr

def load_npy_or_tif(npy_path, tif_path=None):
    if os.path.exists(npy_path): return load_npy(npy_path)
    if tif_path and os.path.exists(tif_path): return load_tif(tif_path)
    return None


# ============================================================
# 一、加载数据
# ============================================================
print("\n[1/4] 加载数据...")

nodata_mask = load_npy(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)
valid_mask  = ~nodata_mask

# ★ v6.0：优先读取最优模型输出
vuln_score = load_best(
    os.path.join(RISK_DIR, 'Risk_Score_Optimal.tif'),
    os.path.join(RISK_DIR, 'Risk_Score_4D.tif'),
    '综合脆弱性得分'
)
risk_level = load_best(
    os.path.join(RISK_DIR, 'Risk_Level_Optimal.tif'),
    os.path.join(RISK_DIR, 'Risk_Level_4D.tif'),
    '脆弱性等级'
)

# 最优模型名称
opt_name_path = os.path.join(RISK_DIR, 'Optimal_Model_Name.txt')
best_model = open(opt_name_path).read().strip() if os.path.exists(opt_name_path) else '未知'
print(f"  最优模型: {best_model}")

# 三维分量
E_score = load_npy_or_tif(os.path.join(RISK_DIR,'Exposure_Score.npy'),
                            os.path.join(RISK_DIR,'Exposure_Score.tif'))
S_score = load_npy_or_tif(os.path.join(RISK_DIR,'Sensitivity_Score.npy'),
                            os.path.join(RISK_DIR,'Sensitivity_Score.tif'))
C_score = load_npy_or_tif(os.path.join(RISK_DIR,'CopingCapacity_Score.npy'),
                            os.path.join(RISK_DIR,'CopingCapacity_Score.tif'))

# 10个原始指标
ind_arrays = {
    '降雨量R':  load_npy(os.path.join(DYN_DIR,  'Precipitation_Mean.npy')),
    '径流CR':   load_npy(os.path.join(DYN_DIR,  'CR_MultiYear_Mean.npy')),
    'TWI':      load_npy(os.path.join(STATIC_DIR,'twi.npy')),
    'HAND(m)':  load_npy(os.path.join(STATIC_DIR,'hand.npy')),
    '积水点WP': load_tif(os.path.join(EXT_DIR,  'waterlogging_point_density_30m.tif')),
    '人口PD':   load_tif(os.path.join(EXT_DIR,  'population_density_30m.tif')),
    '道路RD':   load_tif(os.path.join(EXT_DIR,  'road_density_30m.tif')),
    '避难SH':   load_tif(os.path.join(EXT_DIR,  'shelter_density_30m.tif')),
    '医院HO':   load_tif(os.path.join(EXT_DIR,  'hospital_density_30m.tif')),
    '消防FS':   load_tif(os.path.join(EXT_DIR,  'firestation_density_30m.tif')),
}

# 构建完整有效掩膜
valid_full = valid_mask.copy()
if vuln_score is not None: valid_full &= np.isfinite(vuln_score)
if risk_level is not None: valid_full &= np.isfinite(risk_level)
print(f"  有效像元: {valid_full.sum():,}")


# ============================================================
# 二、★ 构建两个DataFrame（全量KW + 抽样绘图，完全隔离）
# ============================================================
print("\n[2/4] 构建数据集...")

# 全量DataFrame（用于KW检验）
def build_df_full():
    d = {'综合脆弱性': vuln_score[valid_full],
         'Risk_Level':  risk_level[valid_full].astype(int)}
    for name, arr in ind_arrays.items():
        d[name] = arr[valid_full] if arr is not None \
                  else np.full(valid_full.sum(), np.nan, dtype=np.float32)
    if E_score is not None: d['暴露度E'] = E_score[valid_full]
    if S_score is not None: d['敏感性S'] = S_score[valid_full]
    if C_score is not None: d['应对不足C'] = C_score[valid_full]
    df = pd.DataFrame(d).dropna(subset=['综合脆弱性','Risk_Level'])
    return df[df['Risk_Level'].between(1,5)]

df_full = build_df_full()
print(f"  全量DataFrame（KW检验用）: {len(df_full):,} 像元")

# 分析列
kw_cols = [c for c in list(ind_arrays.keys()) + ['暴露度E','敏感性S','应对不足C']
           if c in df_full.columns]


# ============================================================
# 三、Kruskal-Wallis 非参数检验（★ 全量数据）
# ============================================================
print("\n[3/4] Kruskal-Wallis 检验（全量数据，保证学术严谨性）...")
print(f"  {'指标':<14}{'H统计量':>10}{'P值':>12}{'显著性':>8}{'样本量':>10}")
print("  " + "-"*56)

kw_results = {}
for col in kw_cols:
    groups = [df_full[df_full['Risk_Level']==lv][col].dropna().values
              for lv in range(1,6)]
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) < 2:
        kw_results[col] = {'H': np.nan, 'P': np.nan, 'n': 0}
        continue
    h_stat, p_val = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    kw_results[col] = {'H': h_stat, 'P': p_val, 'n': n_total}
    sig = '***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else 'ns'))
    print(f"  {col:<14}{h_stat:>10.2f}{p_val:>12.2e}{sig:>8}{n_total:>10,}")

# 保存KW结果
kw_df = pd.DataFrame([{'指标':k,'H统计量':v['H'],'P值':v['P'],'样本量':v['n'],
                         '显著性':'***' if v['P']<0.001 else ('**' if v['P']<0.01
                         else ('*' if v['P']<0.05 else 'ns')) if not np.isnan(v['P']) else 'NaN'}
                        for k, v in kw_results.items()])
kw_df.to_csv(os.path.join(STAT_DIR,'VSC_KW_Results.csv'),
             index=False, encoding='utf-8-sig')
print(f"  ✅ VSC_KW_Results.csv（含全量样本量）")


# ============================================================
# 四、★ 小提琴图（绘图内部抽样，与KW完全隔离）
# ============================================================
print(f"\n[4/4] 绘制小提琴图（每等级抽样≤{MAX_SAMPLES}，KW结论来自全量）...")

def violin_panel(cols_to_plot, df_full, kw_results, title, out_path, nrows, ncols):
    n_actual = len(cols_to_plot)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols*4.5, nrows*5.5),
                              facecolor='white')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.99)
    # BUG-1修复：nrows=1时subplots返回1D数组，nrows=ncols=1时返回单个Axes，
    # 统一用np.atleast_1d展平，确保axes_list始终是可下标访问的列表
    axes_list = list(np.array(axes).flat)

    # 准备全局图例的句柄（稍后使用）
    global_patches = [mpatches.Patch(facecolor=c, alpha=0.75, label=n)
                      for c, n in zip(LEVEL_COLORS, LEVEL_NAMES)]

    for idx, col in enumerate(cols_to_plot):
        ax = axes_list[idx]

        # ★ 绘图前抽样（不变）
        plot_data = []
        sample_sizes = []
        for lv in range(1, 6):
            vals = df_full[df_full['Risk_Level'] == lv][col].dropna().values
            n_orig = len(vals)
            if n_orig > MAX_SAMPLES:
                vals = np.random.choice(vals, MAX_SAMPLES, replace=False)
            sample_sizes.append(n_orig)
            plot_data.append(vals)

        # 绘制小提琴图（不变）
        if all(len(g) >= 5 for g in plot_data):
            parts = ax.violinplot(
                plot_data, positions=range(1, 6),
                showmeans=False, showmedians=True, showextrema=True
            )
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(LEVEL_COLORS[i])
                pc.set_alpha(0.75)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2.0)
            parts['cmaxes'].set_color('grey')
            parts['cmins'].set_color('grey')

        # ★ 原 KW 标注文字已删除，不再添加

        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(['极低','低','中','高','极高'],
                            rotation=20, fontsize=7.5, ha='right')
        ax.set_title(f'【{col}】', fontsize=10, fontweight='bold')
        ax.set_ylabel(col, fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # ★ 不再在每个子图内单独画图例

    # 隐藏多余子图（不变）
    for idx_hide in range(n_actual, len(axes_list)):
        axes_list[idx_hide].set_visible(False)

    # ★ 在整个图形右侧添加统一的图例
    fig.legend(handles=global_patches, loc='center right',
               fontsize=8, framealpha=0.9,
               title='风险等级', title_fontsize=9,
               bbox_to_anchor=(1.02, 0.5))      # 紧贴右侧居中

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])    # 右侧留出空间给图例
    return fig


# 图5a：10原始指标（2×5布局）
np.random.seed(2024)
raw_cols = [c for c in list(ind_arrays.keys()) if c in df_full.columns]
out_5a = os.path.join(VIS_DIR, 'Step5_KW_Violin_Raw.png')
fig5a = violin_panel(
    raw_cols, df_full, kw_results,
    title=(f'VSC内涝风险10指标分布差异诊断（{best_model}模型 | 北京市 2012-2024）\n'
           f'★ KW检验基于全量数据，小提琴图每等级抽样≤{MAX_SAMPLES}加速渲染'),
    out_path=out_5a, nrows=2, ncols=5
)
fig5a.savefig(out_5a, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Step5_KW_Violin_Raw.png")

# 图5b：E/S/C三维分量（1×3布局）
dim_cols = [c for c in ['暴露度E', '敏感性S', '应对不足C'] if c in df_full.columns]
if dim_cols:
    out_5b = os.path.join(VIS_DIR, 'Step5_KW_Violin_ESC.png')
    fig5b = violin_panel(
        dim_cols, df_full, kw_results,
        title=f'VSC三维分量（E/S/C）在不同风险等级下的分布差异（{best_model}模型）',
        out_path=out_5b, nrows=1, ncols=3
    )
    fig5b.savefig(out_5b, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Step5_KW_Violin_ESC.png")

# ============================================================
# 五、致脆类型识别（8种）
# ============================================================
print("\n  致脆类型识别...")

if all(c in df_full.columns for c in ['暴露度E','敏感性S','应对不足C']):
    E_thr = float(df_full['暴露度E'].median())
    S_thr = float(df_full['敏感性S'].median())
    C_thr = float(df_full['应对不足C'].median())
    print(f"  阈值: E={E_thr:.4f}  S={S_thr:.4f}  C={C_thr:.4f}")

    df_full['E_flag'] = (df_full['暴露度E']   > E_thr).astype(int)
    df_full['S_flag'] = (df_full['敏感性S']   > S_thr).astype(int)
    df_full['C_flag'] = (df_full['应对不足C'] > C_thr).astype(int)

    TYPE_ORDER  = ['O（弱综合型）','E（暴露致险）','S（敏感致险）','C（应对不足）',
                   'ES（暴露-敏感）','EC（暴露-应对）','SC（敏感-应对）','ESC（强综合型）']
    TYPE_COLORS = ['#95A5A6','#E74C3C','#F39C12','#3498DB',
                   '#8E44AD','#E67E22','#2ECC71','#1ABC9C']

    def classify(row):
        e, s, c = row['E_flag'], row['S_flag'], row['C_flag']
        n = e + s + c
        if n==0: return 'O（弱综合型）'
        if n==1:
            if e: return 'E（暴露致险）'
            if s: return 'S（敏感致险）'
            return 'C（应对不足）'
        if n==2:
            if not e: return 'SC（敏感-应对）'
            if not s: return 'EC（暴露-应对）'
            return 'ES（暴露-敏感）'
        return 'ESC（强综合型）'

    df_full['VulnType'] = df_full.apply(classify, axis=1)

    # 统计
    ts = df_full.groupby('VulnType').agg(
        像元数=('综合脆弱性','count'),
        脆弱性均值=('综合脆弱性','mean'),
        脆弱性Std=('综合脆弱性','std')
    ).reset_index()
    ts['面积占比%'] = ts['像元数'] / len(df_full) * 100
    ts = ts.sort_values('面积占比%', ascending=False)
    print(f"\n  致脆类型统计:")
    print(ts[['VulnType','像元数','面积占比%','脆弱性均值']].to_string(
          index=False, float_format='{:.4f}'.format))

    ts.to_csv(os.path.join(STAT_DIR,'VSC_VulnType_Statistics.csv'),
              index=False, encoding='utf-8-sig')

    lv_type = pd.crosstab(df_full['Risk_Level'], df_full['VulnType'],
                           normalize='index') * 100
    lv_type.round(2).to_csv(os.path.join(STAT_DIR,'VSC_LevelType_Crosstab.csv'),
                              encoding='utf-8-sig')
    print(f"  ✅ VSC_VulnType_Statistics.csv + VSC_LevelType_Crosstab.csv")

    # 致脆类型图（饼图 + 堆叠柱）
    fig6, axes6 = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig6.suptitle(f'北京市内涝风险致险类型分析（{best_model}模型，VSC三维框架）',
                   fontsize=13, fontweight='bold')

    cnts = [df_full['VulnType'].value_counts().get(t,0) for t in TYPE_ORDER]
    valid_t = [(t,c,col) for t,c,col in zip(TYPE_ORDER,cnts,TYPE_COLORS) if c>0]
    if valid_t:
        tl, tc, tco = zip(*valid_t)
        axes6[0].pie(tc, labels=None, colors=tco, autopct='%1.1f%%',
                      startangle=90, pctdistance=0.78,
                      wedgeprops={'linewidth':1,'edgecolor':'white'})
        pp = [mpatches.Patch(color=c, label=t) for t,c in zip(tl,tco)]
        axes6[0].legend(handles=pp, loc='lower left', fontsize=7.5,
                         framealpha=0.9, bbox_to_anchor=(-0.05,-0.1))
        axes6[0].set_title('全域致脆类型面积占比', fontsize=12, fontweight='bold')

    lv_arr = np.zeros((5, len(TYPE_ORDER)))
    for ki, t in enumerate(TYPE_ORDER):
        for lv in range(1,6):
            mk  = df_full['Risk_Level'] == lv
            tot = mk.sum()
            lv_arr[lv-1,ki] = (df_full.loc[mk,'VulnType']==t).sum()/tot*100 if tot>0 else 0

    bot = np.zeros(5)
    for ki,(t,col) in enumerate(zip(TYPE_ORDER,TYPE_COLORS)):
        vals = lv_arr[:,ki]
        if vals.max()>0.5:
            axes6[1].bar(range(1,6), vals, bottom=bot, color=col,
                          alpha=0.85, label=t, edgecolor='white')
        bot += vals

    axes6[1].set_xticks(range(1,6))
    axes6[1].set_xticklabels(LEVEL_NAMES, rotation=20, fontsize=9, ha='right')
    axes6[1].set_ylabel('类型占比 (%)', fontsize=11)
    axes6[1].set_ylim(0,105)
    axes6[1].set_title('各风险等级内致险类型构成', fontsize=12, fontweight='bold')
    axes6[1].legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
    axes6[1].grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.96])
    out_6 = os.path.join(VIS_DIR,'Step5_VulnType_Analysis.png')
    fig6.savefig(out_6, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Step5_VulnType_Analysis.png")

else:
    print(f"  ⚠️  E/S/C分量未找到，跳过致脆类型识别")


# ============================================================
# 六、统计汇总表
# ============================================================
print("\n  生成统计汇总表...")

summary_cols = [c for c in list(ind_arrays.keys())+['暴露度E','敏感性S','应对不足C','综合脆弱性']
                if c in df_full.columns]
rows = []
for lv in range(1,6):
    mk  = df_full['Risk_Level']==lv
    row = {'脆弱性等级': LEVEL_NAMES[lv-1], '像元数': int(mk.sum())}
    for col in summary_cols:
        v = df_full.loc[mk, col].dropna().values
        row[f'{col}_mean'] = float(np.mean(v)) if len(v)>0 else np.nan
        row[f'{col}_std']  = float(np.std(v))  if len(v)>0 else np.nan
    rows.append(row)

pd.DataFrame(rows).to_csv(
    os.path.join(STAT_DIR,'VSC_Risk_Driver_Summary.csv'),
    index=False, encoding='utf-8-sig')
print(f"  ✅ VSC_Risk_Driver_Summary.csv")

print(f"\n{'='*70}")
print(f"Step 5 (v6.0) 完成！")
print(f"  最优模型: {best_model}")
print(f"  📊 {VIS_DIR}/")
print(f"     Step5_KW_Violin_Raw.png   ← 10指标（抽样≤{MAX_SAMPLES}/等级）")
print(f"     Step5_KW_Violin_ESC.png   ← E/S/C分量")
print(f"     Step5_VulnType_Analysis.png")
print(f"  📋 {STAT_DIR}/")
print(f"     VSC_KW_Results.csv（全量样本量标注）")
print(f"     VSC_Risk_Driver_Summary.csv")
print(f"     VSC_VulnType_Statistics.csv + VSC_LevelType_Crosstab.csv")
print(f"{'='*70}")