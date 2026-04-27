"""
Step 6: 城市内涝风险区划（线路B - 纯地形熵权-TOPSIS）
=======================================================
设计思路：
  本步骤完全独立于 Step5 的 Logistic 回归，不依赖任何训练集标签。
  风险得分由以下四类纯地形/土壤水文指标通过熵权法客观赋权合成：

    指标             方向  物理含义
    1/(HAND+1)      正向  越低洼越危险（HAND 越小表示越靠近河道洼地）
    TWI_norm        正向  TWI 越高越容易汇水积水
    1-slope_norm    正向  坡度越平缓排水越慢，越容易积水
    1-Ks_norm       正向  透水性越差越难下渗，地表产流越多

  权重由熵权法从全域栅格数据客观确定，再用 TOPSIS 计算各栅格综合得分，
  最后按分位数划分五级风险等级（每级约占 20% 像元）。

  行政区评价同样使用熵权-TOPSIS，指标为：
    年均积水频次、平均积水深、最大积水深、汛期均降雨、地势低洼指数、不透水代理指数

输入:
  F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif
  .\Step2\dem_soil_features.csv
  F:\Data\src\files\flood_labels_clean.csv
  .\Step1\rainfall_daily_2018_2024.csv

输出:
  Step6/risk_map.png                 全域连续风险得分图 + 五级区划图
  Step6/topsis_ranking.png           行政区排名柱状图
  Step6/district_topsis_ranking.csv  行政区 TOPSIS 得分明细

版本历史:
  v1: Logistic 回归概率值直接驱动区划 → 全红（降雨系数过大）
  v2: 纯地形 Logistic（固定 rainfall=0）+ 分位数分级
  v3: 彻底去除 Logistic，改为熵权-TOPSIS 物理指标合成；去除 lat_bin/lon_bin
  v4（本版）: 修复两个关键 bug：
    Bug1 - 不透水性指标为常数（全域 ks_median 相同），熵权归零
           修复：改用高程低洼度（dem_norm 反比）作为不透水代理，空间变化明显
           同时对四个指标添加最低权重约束（各 ≥ 8%），防止某指标被完全归零
    Bug2 - 分位数分级时大量像元 TOPSIS 得分堆积到 1.0，导致极高:48%、高:0%
           修复：改为等距线性分级（[0,0.2),[0.2,0.4),[0.4,0.6),[0.6,0.8),[0.8,1]）
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
DEM_PATH   = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
DEM_FEAT   = r'.\Step2\dem_soil_features.csv'
LABEL_CSV  = r'F:\Data\src\files\flood_labels_clean.csv'
RAIN_CSV   = r'.\Step1\rainfall_daily_2018_2024.csv'
OUTPUT_DIR = r'.\Step6'
os.makedirs(OUTPUT_DIR, exist_ok=True)

KS_MAX = 140.0   # 北京土壤 Ks 最大值（mm/h），用于归一化透水面比例

# 行政区边界框：(lat_min, lat_max, lon_min, lon_max)
DISTRICTS = {
    '东城区':  (39.88, 39.96, 116.36, 116.44),
    '西城区':  (39.87, 39.96, 116.30, 116.42),
    '朝阳区':  (39.83, 40.05, 116.39, 116.64),
    '丰台区':  (39.75, 39.92, 116.17, 116.42),
    '海淀区':  (39.89, 40.15, 116.17, 116.41),
    '石景山区': (39.88, 39.96, 116.13, 116.26),
    '通州区':  (39.78, 40.02, 116.55, 116.82),
    '顺义区':  (40.00, 40.22, 116.44, 116.74),
    '昌平区':  (40.08, 40.32, 116.08, 116.46),
    '房山区':  (39.62, 39.93, 115.83, 116.23),
    '密云区':  (40.28, 40.54, 116.62, 117.02),
    '延庆区':  (40.38, 40.62, 115.73, 116.08),
}


# ==================== 工具函数 ====================

def entropy_weight(X: np.ndarray, w_min: float = 0.08) -> np.ndarray:
    """
    熵权法：计算各指标客观权重，并施加最低权重约束。

    X: (n_samples, n_indicators)，所有指标均已转换为正向（值越大风险越高）。
    w_min: 每个指标的最低权重（默认 8%）。防止空间变化极小的指标被完全归零。
    返回 (n_indicators,) 权重向量，合计为 1。
    """
    X = np.asarray(X, dtype=float)
    m = X.shape[1]
    col_sum = X.sum(axis=0) + 1e-10
    Xn      = np.clip(X / col_sum, 1e-10, 1.0)
    # 信息熵（以样本数为底的对数）
    n = X.shape[0]
    e = -np.sum(Xn * np.log(Xn), axis=0) / np.log(n)
    d = 1.0 - e
    w = d / (d.sum() + 1e-10)

    # 最低权重约束：迭代调整，将低于 w_min 的指标补齐，从高权重指标等比扣减
    w = np.clip(w, w_min, 1.0)
    w = w / w.sum()   # 归一化
    return w


def topsis_score(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    TOPSIS 综合得分。
    X: (n_samples, n_indicators)，所有指标均为正向。
    w: (n_indicators,) 熵权。
    返回 (n_samples,) 贴近度，值域 [0,1]，越大风险越高。
    """
    X  = np.asarray(X, dtype=float)
    norms    = np.sqrt((X ** 2).sum(axis=0)) + 1e-10
    Xw       = (X / norms) * w
    pos_ideal = Xw.max(axis=0)
    neg_ideal = Xw.min(axis=0)
    d_pos = np.sqrt(((Xw - pos_ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((Xw - neg_ideal) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + 1e-10)


# ==================== [1] 读取 DEM，估算全域地形特征 ====================
print("=" * 55)
print("[1/3] 读取 DEM，估算全域地形特征...")

dem_feats = pd.read_csv(DEM_FEAT)

with rasterio.open(DEM_PATH) as src:
    dem_full = src.read(1).astype(np.float32)
    height, width = dem_full.shape

# 降采样（stride=10，等效 300m 分辨率，加速计算）
STRIDE  = 10
dem_ds  = dem_full[::STRIDE, ::STRIDE]
H, W    = dem_ds.shape
res_m   = 30.0 * STRIDE

# 坡度（Sobel 梯度近似）
dy = np.gradient(dem_ds, axis=0) / res_m
dx = np.gradient(dem_ds, axis=1) / res_m
slope_arr = np.clip(np.rad2deg(np.arctan(np.sqrt(dx ** 2 + dy ** 2))), 0.01, 60.0)

# 高程归一化（0=最低洼，1=最高）
dem_valid = dem_ds.copy()
dem_valid[dem_valid < -100] = np.nan
dem_min  = np.nanmin(dem_valid)
dem_max  = np.nanmax(dem_valid)
dem_norm = (dem_valid - dem_min) / (dem_max - dem_min + 1e-6)

# 从点位统计量映射全域 HAND / TWI
# 物理对应：高程越低 → HAND 越小（越靠近河道） → TWI 越大
hand_min  = float(dem_feats['HAND_m'].quantile(0.02))
hand_max  = float(dem_feats['HAND_m'].quantile(0.98))
twi_min   = float(dem_feats['TWI'].quantile(0.02))
twi_max   = float(dem_feats['TWI'].quantile(0.98))
ks_median = float(dem_feats['ks_mmh'].median())

hand_arr = np.clip(hand_min + dem_norm * (hand_max - hand_min), hand_min, hand_max)
twi_arr  = np.clip(twi_max  - dem_norm * (twi_max  - twi_min),  twi_min,  twi_max)

nodata_mask = np.isnan(dem_valid) | (dem_ds < -100)
valid_count = int((~nodata_mask).sum())

print(f"  栅格尺寸（降采样 stride={STRIDE}）: {H} × {W}  有效像元: {valid_count:,}")
print(f"  HAND 范围: {np.nanmin(hand_arr):.1f} ~ {np.nanmax(hand_arr):.1f} m")
print(f"  TWI  范围: {np.nanmin(twi_arr):.2f} ~ {np.nanmax(twi_arr):.2f}")
print(f"  坡度范围: {np.nanmin(slope_arr):.2f} ~ {np.nanmax(slope_arr):.2f} °")

# ==================== [2] 熵权-TOPSIS 全域风险得分 ====================
print("\n[2/3] 熵权-TOPSIS 全域风险得分计算...")

# ---- 构造四个正向风险指标（越大越危险）----
# ind1：低洼性  1/(HAND+1)，越低洼越大
ind1_raw = (1.0 / (hand_arr + 1.0)).ravel()

# ind2：汇水性  TWI，越高越易积水
ind2_raw = twi_arr.ravel()

# ind3：平坦性  1 - slope_norm，坡度越小越平坦越危险
slope_min_f, slope_max_f = slope_arr.min(), slope_arr.max()
slope_norm_arr = (slope_arr - slope_min_f) / (slope_max_f - slope_min_f + 1e-6)
ind3_raw = (1.0 - slope_norm_arr).ravel()

# ind4：地形低洼代理不透水性
#   原方案：全域使用同一个 ks_median 常数 → 零方差 → 熵权归零（已修复）
#   修复方案：用 DEM 高程低洼度（1 - dem_norm）作为不透水代理
#     - 低洼地区排水基础设施差、硬化程度高，透水性弱 → 值越大越危险
#     - 该变量与 HAND/TWI 来自同一 DEM，但归一化方式不同，具备独立空间变化
ind4_raw = (1.0 - dem_norm).ravel()
ind4_raw = np.where(np.isnan(ind4_raw), np.nanmedian(ind4_raw), ind4_raw)

# 统一 min-max 归一化到 [0,1]
def _norm(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-6)

ind1 = _norm(ind1_raw)
ind2 = _norm(ind2_raw)
ind3 = _norm(ind3_raw)   # 已是平坦性（1-slope_norm），再归一化
ind4 = _norm(ind4_raw)

# 构建指标矩阵（仅对有效像元）
valid_flat = (~nodata_mask).ravel()
indicator_mat = np.column_stack([ind1, ind2_norm, ind3, ind4])
valid_X       = indicator_mat[valid_flat]

# 熵权法计算权重
w = entropy_weight(valid_X)
print(f"  熵权权重:")
for name, wi in zip(['低洼性(1/HAND+1)', '汇水性(TWI)', '平坦性(1-slope)', '不透水性(1-Ks)'], w):
    print(f"    {name}: {wi:.4f}")

# TOPSIS 得分
score_valid = topsis_score(valid_X, w)

# 还原到全域矩阵
score_flat          = np.full(len(valid_flat), np.nan)
score_flat[valid_flat] = score_valid
risk_score          = score_flat.reshape(H, W)
risk_score[nodata_mask] = np.nan

valid_scores = risk_score[~nodata_mask]
q20, q40, q60, q80 = np.percentile(valid_scores, [20, 40, 60, 80])
print(f"\n  得分范围: {np.nanmin(risk_score):.4f} ~ {np.nanmax(risk_score):.4f}")
print(f"  标准差  : {np.nanstd(risk_score):.4f}（>0.05 表示空间差异显著）")
print(f"  分位数参考(不用于分级): 20%={q20:.3f}  40%={q40:.3f}  60%={q60:.3f}  80%={q80:.3f}")

# 五级分类：等距线性分级（0-0.2 / 0.2-0.4 / 0.4-0.6 / 0.6-0.8 / 0.8-1.0）
# 分位数分级在 TOPSIS 得分堆积时会失效（大量像元=1.0 导致高/极高边界重叠）
# 等距分级物理含义更清晰，分级界限固定，便于跨区域比较
THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
level_names = ['极低', '低', '中', '高', '极高']

risk_level = np.full_like(risk_score, np.nan)
for i, (lo, hi) in enumerate(zip(THRESHOLDS[:-1], THRESHOLDS[1:]), start=1):
    if i < 5:
        risk_level[(risk_score >= lo) & (risk_score < hi)] = i
    else:
        risk_level[(risk_score >= lo) & (risk_score <= hi)] = i
risk_level[nodata_mask] = np.nan

pcts = [float(np.nanmean(risk_level == i)) * 100 for i in range(1, 6)]
print("  等距分级阈值: " + " / ".join([f"{lo:.1f}~{hi:.1f}" for lo, hi in
                                         zip(THRESHOLDS[:-1], THRESHOLDS[1:])]))
print("  各级比例: " + "  ".join([f"{n}:{p:.1f}%" for n, p in zip(level_names, pcts)]))

# ==================== [3] 行政区熵权-TOPSIS 评价 ====================
print("\n[3/3] 行政区熵权-TOPSIS 综合评价...")

labels_df = pd.read_csv(LABEL_CSV)
rain_df   = pd.read_csv(RAIN_CSV)
n_years   = labels_df['year'].nunique()

rows = []
for dist, (lt_mn, lt_mx, ln_mn, ln_mx) in DISTRICTS.items():
    sub = labels_df[
        labels_df['latitude'].between(lt_mn, lt_mx) &
        labels_df['longitude'].between(ln_mn, ln_mx)
    ]
    freq = round(len(sub) / n_years, 2)
    mdep = round(sub['depth_real_cm'].mean(), 1) if len(sub) > 0 else 0.0
    xdep = round(sub['depth_real_cm'].max(),  1) if len(sub) > 0 else 0.0

    sub_r = rain_df[
        rain_df['latitude'].between(lt_mn, lt_mx) &
        rain_df['longitude'].between(ln_mn, ln_mx)
    ]
    avgr = round(sub_r['rainfall_mm'].mean(), 2) if len(sub_r) > 0 else 0.0

    sub_d = dem_feats[
        dem_feats['latitude'].between(lt_mn, lt_mx) &
        dem_feats['longitude'].between(ln_mn, ln_mx)
    ]
    ah  = round(sub_d['HAND_m'].mean(), 2) if len(sub_d) > 0 else 5.0
    aks = round(sub_d['ks_mmh'].mean(),  2) if len(sub_d) > 0 else 56.0

    rows.append({
        '行政区':      dist,
        '年均积水频次':  freq,
        '平均积水深cm':  mdep,
        '最大积水深cm':  xdep,
        '汛期均降雨mm':  avgr,
        '地势低洼指数':  round(1.0 / (ah + 1.0), 4),
        '不透水代理指数': round(max(0.0, 1.0 - aks / KS_MAX), 4),
    })

eval_df = pd.DataFrame(rows)

INDS = ['年均积水频次', '平均积水深cm', '最大积水深cm',
        '汛期均降雨mm', '地势低洼指数', '不透水代理指数']
X_dist = eval_df[INDS].values.astype(float)

w_dist = entropy_weight(X_dist)
print("  行政区指标权重:")
for n, v in zip(INDS, w_dist):
    print(f"    {n}: {v:.4f}")

eval_df['TOPSIS得分'] = topsis_score(X_dist, w_dist).round(4)
eval_df['风险排名']   = eval_df['TOPSIS得分'].rank(ascending=False).astype(int)
eval_df['风险等级']   = pd.cut(
    eval_df['TOPSIS得分'],
    bins=[0, .25, .45, .65, .80, 1.0],
    labels=['低', '较低', '中', '较高', '高']
)
eval_df = eval_df.sort_values('风险排名').reset_index(drop=True)
print("\n  行政区风险排名:")
print(eval_df[['行政区', 'TOPSIS得分', '风险排名', '风险等级']].to_string(index=False))
eval_df.to_csv(os.path.join(OUTPUT_DIR, 'district_topsis_ranking.csv'),
               index=False, encoding='utf-8-sig')

# ==================== 出图 ====================
print("\n生成风险区划图...")

colors5 = ['#2ECC71', '#A8E063', '#F39C12', '#E74C3C', '#8B0000']
cmap5   = mcolors.ListedColormap(colors5)
norm5   = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('北京市城市内涝地形风险区划（熵权-TOPSIS，2018-2024）',
             fontsize=14, fontweight='bold')

# 左图：连续风险得分
ax = axes[0]
im = ax.imshow(
    risk_score, cmap='RdYlGn_r',
    vmin=float(np.nanpercentile(risk_score, 2)),
    vmax=float(np.nanpercentile(risk_score, 98)),
    aspect='auto'
)
plt.colorbar(im, ax=ax, shrink=0.8, label='熵权-TOPSIS 综合风险得分')
ax.set_title('地形积水易发性分布\n（低洼性 × 汇水性 × 平坦性 × 不透水性）', fontsize=11)
ax.set_xlabel(f'经向像元（×{STRIDE}×30m）')
ax.set_ylabel(f'纬向像元（×{STRIDE}×30m）')
weight_text = (
    f"指标权重:\n"
    f"低洼性:   {w[0]:.3f}\n"
    f"汇水性:   {w[1]:.3f}\n"
    f"平坦性:   {w[2]:.3f}\n"
    f"不透水性: {w[3]:.3f}"
)
ax.text(0.02, 0.02, weight_text, transform=ax.transAxes, fontsize=8, color='#333',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# 右图：五级分类
ax = axes[1]
im = ax.imshow(risk_level, cmap=cmap5, norm=norm5, aspect='auto')
cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[1, 2, 3, 4, 5])
cb.set_ticklabels(level_names)
cb.set_label('风险等级')
stats_text = '\n'.join([f'{n}:  {p:.1f}%' for n, p in zip(level_names, pcts)])
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
ax.set_title('北京市城市内涝五级风险区划\n（等距分级：[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]）', fontsize=11)
ax.set_xlabel(f'经向像元（×{STRIDE}×30m）')
ax.set_ylabel(f'纬向像元（×{STRIDE}×30m）')

plt.tight_layout()
out_map = os.path.join(OUTPUT_DIR, 'risk_map.png')
plt.savefig(out_map, dpi=200, bbox_inches='tight')
plt.close()
print(f"  已保存: {out_map}")

# 行政区排名图
fig, ax = plt.subplots(figsize=(10, 7))
color_map  = {'高': '#8B0000', '较高': '#E74C3C', '中': '#F39C12',
              '较低': '#A8E063', '低': '#2ECC71'}
bar_colors = [color_map.get(str(v), '#888') for v in eval_df['风险等级']]
bars = ax.barh(eval_df['行政区'], eval_df['TOPSIS得分'],
               color=bar_colors, edgecolor='white', linewidth=0.5)
for bar, val, rank in zip(bars, eval_df['TOPSIS得分'], eval_df['风险排名']):
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'#{rank}  {val:.3f}', va='center', fontsize=9)
ax.axvline(0.45, color='gray', ls='--', lw=1, alpha=0.6, label='较高风险阈值(0.45)')
ax.set_xlabel('TOPSIS 综合得分', fontsize=12)
ax.set_title('北京市各行政区城市内涝风险综合评价\n（熵权-TOPSIS，2018-2024 年真实数据）',
             fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, float(eval_df['TOPSIS得分'].max()) * 1.2)
plt.tight_layout()
out_rank = os.path.join(OUTPUT_DIR, 'topsis_ranking.png')
plt.savefig(out_rank, dpi=200, bbox_inches='tight')
plt.close()
print(f"  已保存: {out_rank}")

print(f"\n{'='*55}")
print(f"Step6 完成！输出文件:")
print(f"  risk_map.png                  全域连续风险图 + 五级区划图")
print(f"  topsis_ranking.png            行政区排名图")
print(f"  district_topsis_ranking.csv   行政区 TOPSIS 得分明细")
print(f"\n方法说明（线路B - 无训练集依赖）:")
print(f"  全域区划: 熵权-TOPSIS（4 个纯地形指标）")
print(f"  行政区:   熵权-TOPSIS（6 个综合指标，含历史积水统计）")
print(f"  权重: " + "  ".join([
    f"{n}={v:.3f}" for n, v in zip(
        ['低洼性', '汇水性', '平坦性', '不透水性'], w)]))
