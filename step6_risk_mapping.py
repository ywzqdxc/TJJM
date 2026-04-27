"""
Step 6: 内涝风险区划图 + 熵权-TOPSIS行政区评价（最终版）
=========================================================
修复：
  1. 路径更新为实际目录结构
  2. 特征列表与Step5一致（7个，已去掉VIF>10的特征）
  3. 汛期降雨量用真实Step1数据，不透水面积比用Ks代理
  4. 模型用2018-2023训练，2024为测试集
  5. 评价指标全部来自真实数据，无随机填充

输出:
  Step6/risk_map.png              全市风险概率+五级区划图
  Step6/topsis_ranking.png        行政区排名图
  Step6/district_topsis_ranking.csv  TOPSIS评价数据
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import rasterio
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
DATASET_CSV = r'.\Step4\All_Years_Dataset.csv'
LABEL_CSV   = r'F:\Data\src\files\flood_labels_clean.csv'
RAIN_CSV    = r'.\Step1\rainfall_daily_2018_2024.csv'
DEM_PATH    = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
DEM_FEAT    = r'.\Step2\dem_soil_features.csv'
OUTPUT_DIR  = r'.\Step6'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step5一致的7个特征（VIF<10）
FEATURES = ['rainfall_mm','RE_mm','HAND_m','TWI','slope_deg','dem_m','theta_s']

# ==================== [1] 训练模型 ====================
print("[1/4] 训练空间Logistic模型（2018-2023）...")
df = pd.read_csv(DATASET_CSV)

train = df[df['year'].between(2018,2023)].copy()
train['lat_bin'] = pd.cut(train['latitude'],  bins=5, labels=False)
train['lon_bin'] = pd.cut(train['longitude'], bins=5, labels=False)
feat_use = FEATURES + ['lat_bin','lon_bin']

X_tr = train[feat_use].fillna(0).values
y_tr = train['label'].values

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_tr_s, y_tr)
print(f"  训练完成：{len(train)}条样本（正{y_tr.sum()}/负{(y_tr==0).sum()}）")

# ==================== [2] 全市栅格风险预测 ====================
print("\n[2/4] 全市风险预测（典型暴雨情景）...")

# 读取DEM
with rasterio.open(DEM_PATH) as src:
    dem_full  = src.read(1).astype(np.float32)
    transform = src.transform
    crs       = src.crs
    height, width = dem_full.shape
    nodata    = src.nodata

print(f"  DEM尺寸: {height}×{width} ({height*width/1e6:.1f}M像元)")

# 降采样（全市30m分辨率太大，用stride=10快速预览）
# 正式出图可改STRIDE=1，但需要30分钟+内存
STRIDE = 10
dem_ds = dem_full[::STRIDE, ::STRIDE]
H, W   = dem_ds.shape
print(f"  降采样后: {H}×{W} ({H*W/1e6:.2f}M像元，stride={STRIDE})")

# 读取积水点位特征作为全市统计参考
dem_feats = pd.read_csv(DEM_FEAT)
theta_s_med = dem_feats['theta_s'].median()
twi_med     = dem_feats['TWI'].median()
hand_med    = dem_feats['HAND_m'].median()

# 典型情景：2023年"7·29"特大暴雨，北京市区日降雨约141mm
SCENARIO_RAIN = 141.0  # mm（历史实测，2023年最大日降雨）

# 基于DEM逐像元估算地形特征
# 坡度用DEM梯度近似（分辨率已知30m）
res_m = 30.0 * STRIDE
dy = np.gradient(dem_ds, axis=0) / res_m
dx = np.gradient(dem_ds, axis=1) / res_m
slope_est = np.clip(np.rad2deg(np.arctan(np.sqrt(dx**2 + dy**2))), 0, 60)

# 高程归一化
dem_valid = dem_ds.copy()
dem_valid[dem_valid < -100] = np.nan
dem_norm  = (dem_valid - np.nanmin(dem_valid)) / (np.nanmax(dem_valid) - np.nanmin(dem_valid) + 1e-6)

# TWI：低洼地（高程低）TWI更高
twi_est  = np.clip(twi_med * (1.0 - dem_norm * 0.35), 4, 20)
# HAND：低地靠近河道，HAND小
hand_est = np.clip(dem_norm * hand_med * 3, 0, 50)

# RE：用修复版Green-Ampt公式（与Step4一致）
KS_MAX = 140.0
# 全市用中位数Ks估算（城区Ks偏小→RE偏高，合理）
ks_med     = dem_feats['ks_mmh'].median()
perv_ratio = min(ks_med / KS_MAX, 1.0)
deficit    = max(0, theta_s_med - 0.25)  # 典型前期SM=0.25
imperv_ru  = max(0.0, SCENARIO_RAIN - 2.0) * (1 - perv_ratio)
f_eff      = ks_med * max(0.3, deficit/0.3)
perv_ru    = max(0.0, SCENARIO_RAIN - f_eff) * perv_ratio
re_est     = imperv_ru + perv_ru

print(f"  情景参数: 日降雨={SCENARIO_RAIN}mm  RE估算={re_est:.1f}mm")

# 构建特征矩阵（7个特征 + 2个区位虚拟变量）
feat_grid = np.zeros((H*W, 9), dtype=np.float32)
feat_grid[:,0] = SCENARIO_RAIN          # rainfall_mm
feat_grid[:,1] = re_est                 # RE_mm（全市统一，后续可逐像元细化）
feat_grid[:,2] = hand_est.ravel()       # HAND_m
feat_grid[:,3] = twi_est.ravel()        # TWI
feat_grid[:,4] = slope_est.ravel()      # slope_deg
feat_grid[:,5] = dem_ds.ravel()         # dem_m
feat_grid[:,6] = theta_s_med           # theta_s
feat_grid[:,7] = 2                      # lat_bin（全市中部）
feat_grid[:,8] = 2                      # lon_bin

# 预测
feat_s = scaler.transform(feat_grid)
proba  = model.predict_proba(feat_s)[:,1].reshape(H, W).astype(np.float32)

# 掩膜无效DEM
nodata_mask = np.isnan(dem_ds) | (dem_ds < -100)
proba[nodata_mask] = np.nan

print(f"  概率范围: {np.nanmin(proba):.3f} ~ {np.nanmax(proba):.3f}")
print(f"  各风险等级像元比例:")
for lo, hi, name in [(0,.2,'极低'),(0.2,.4,'低'),(0.4,.6,'中'),(0.6,.8,'高'),(0.8,1.01,'极高')]:
    pct = np.nanmean((proba>=lo)&(proba<hi))*100
    print(f"    {name}: {pct:.1f}%")

# ==================== [3] 出图 ====================
print("\n[3/4] 生成风险区划图...")

risk_level = np.full_like(proba, np.nan)
risk_level[(proba>=0)   & (proba<0.2)] = 1
risk_level[(proba>=0.2) & (proba<0.4)] = 2
risk_level[(proba>=0.4) & (proba<0.6)] = 3
risk_level[(proba>=0.6) & (proba<0.8)] = 4
risk_level[proba>=0.8]                 = 5

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：连续概率
ax = axes[0]
im1 = ax.imshow(proba, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im1, ax=ax, shrink=0.8, label='积水概率')
ax.set_title(f'北京市内涝风险概率分布\n（情景：2023年"7·29"特大暴雨 {SCENARIO_RAIN}mm）', fontsize=12)
ax.set_xlabel('经向像元')
ax.set_ylabel('纬向像元')
ax.text(0.02, 0.02, '基于空间Logistic回归（2018-2023年训练）',
        transform=ax.transAxes, fontsize=8, color='gray')

# 右图：五级区划
ax = axes[1]
colors5 = ['#2ECC71','#A8E063','#F39C12','#E74C3C','#8B0000']
cmap5   = mcolors.ListedColormap(colors5)
norm5   = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], 5)
im2 = ax.imshow(risk_level, cmap=cmap5, norm=norm5, aspect='auto')
cbar = plt.colorbar(im2, ax=ax, shrink=0.8, ticks=[1,2,3,4,5])
cbar.set_ticklabels(['极低','低','中','高','极高'])
cbar.set_label('风险等级')
ax.set_title('北京市城市内涝五级风险区划\n（空间Logistic回归+Green-Ampt物理特征）', fontsize=12)
ax.set_xlabel('经向像元')
ax.set_ylabel('纬向像元')

# 图例统计
stats = '\n'.join([
    f"极低: {np.nanmean(risk_level==1)*100:.1f}%",
    f"低  : {np.nanmean(risk_level==2)*100:.1f}%",
    f"中  : {np.nanmean(risk_level==3)*100:.1f}%",
    f"高  : {np.nanmean(risk_level==4)*100:.1f}%",
    f"极高: {np.nanmean(risk_level==5)*100:.1f}%",
])
ax.text(0.02, 0.98, stats, transform=ax.transAxes,
        fontsize=8, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
out_risk = os.path.join(OUTPUT_DIR, 'risk_map.png')
plt.savefig(out_risk, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✅ 已保存: {out_risk}")

# ==================== [4] 熵权-TOPSIS ====================
print("\n[4/4] 熵权-TOPSIS行政区综合评价...")

# 读取真实积水数据
labels = pd.read_csv(LABEL_CSV)
rain   = pd.read_csv(RAIN_CSV)

# 北京主要行政区坐标范围
districts = {
    '东城区': (39.88, 39.96, 116.36, 116.44),
    '西城区': (39.87, 39.96, 116.30, 116.42),
    '朝阳区': (39.83, 40.05, 116.39, 116.64),
    '丰台区': (39.75, 39.92, 116.17, 116.42),
    '海淀区': (39.89, 40.15, 116.17, 116.41),
    '石景山区':(39.88, 39.96, 116.13, 116.26),
    '通州区': (39.78, 40.02, 116.55, 116.82),
    '顺义区': (40.00, 40.22, 116.44, 116.74),
    '昌平区': (40.08, 40.32, 116.08, 116.46),
    '房山区': (39.62, 39.93, 115.83, 116.23),
    '密云区': (40.28, 40.54, 116.62, 117.02),
    '延庆区': (40.38, 40.62, 115.73, 116.08),
}

eval_rows = []
for dist, (lat_min, lat_max, lon_min, lon_max) in districts.items():
    # 该区历史积水数据
    sub = labels[
        labels['latitude'].between(lat_min, lat_max) &
        labels['longitude'].between(lon_min, lon_max)
    ]
    n_events  = len(sub)
    n_years   = labels['year'].nunique()
    freq      = round(n_events / n_years, 2)
    mean_dep  = round(sub['depth_real_cm'].mean(), 1) if n_events > 0 else 0.0
    max_dep   = round(sub['depth_real_cm'].max(),  1) if n_events > 0 else 0.0

    # 该区汛期平均日降雨（来自Step1真实数据）
    sub_rain = rain[
        rain['latitude'].between(lat_min, lat_max) &
        rain['longitude'].between(lon_min, lon_max)
    ]
    avg_rain = round(sub_rain['rainfall_mm'].mean(), 2) if len(sub_rain) > 0 else 0.0

    # 该区积水点位平均HAND（地势低洼程度代理）
    sub_dem = dem_feats[
        dem_feats['latitude'].between(lat_min, lat_max) &
        dem_feats['longitude'].between(lon_min, lon_max)
    ]
    avg_hand = round(sub_dem['HAND_m'].mean(), 2) if len(sub_dem) > 0 else 5.0
    avg_ks   = round(sub_dem['ks_mmh'].mean(),  2) if len(sub_dem) > 0 else 56.0
    # Ks越小→不透水率越高→风险越高；转换为不透水代理指数（越大越危险）
    imperv_idx = round(max(0, 1 - avg_ks / KS_MAX), 4)

    eval_rows.append({
        '行政区'    : dist,
        '年均积水频次': freq,
        '平均积水深cm': mean_dep,
        '最大积水深cm': max_dep,
        '汛期均降雨mm': avg_rain,
        '地势低洼指数': round(1 / (avg_hand + 1), 4),  # HAND越小→越低洼→越危险
        '不透水代理指数': imperv_idx,
    })

eval_df = pd.DataFrame(eval_rows)
print("\n  各行政区原始评价指标:")
print(eval_df.to_string(index=False))

# 熵权法
INDICATORS = ['年均积水频次','平均积水深cm','最大积水深cm',
              '汛期均降雨mm','地势低洼指数','不透水代理指数']

def entropy_weight(X):
    X = X.astype(float)
    col_sum = X.sum(axis=0) + 1e-10
    X_norm  = X / col_sum
    X_norm  = np.clip(X_norm, 1e-10, 1)
    entropy = -np.sum(X_norm * np.log(X_norm), axis=0) / np.log(len(X))
    w       = (1 - entropy) / (np.sum(1 - entropy) + 1e-10)
    return w

X_eval = eval_df[INDICATORS].values
weights = entropy_weight(X_eval)
print(f"\n  熵权法权重:")
for n, w in zip(INDICATORS, weights):
    print(f"    {n}: {w:.4f}")

# TOPSIS
def topsis(X, w):
    X = X.astype(float)
    norm = np.sqrt((X**2).sum(axis=0)) + 1e-10
    Xw   = (X / norm) * w
    pos  = Xw.max(axis=0)
    neg  = Xw.min(axis=0)
    d_p  = np.sqrt(((Xw - pos)**2).sum(axis=1))
    d_n  = np.sqrt(((Xw - neg)**2).sum(axis=1))
    return d_n / (d_p + d_n + 1e-10)

scores = topsis(X_eval, weights)
eval_df['TOPSIS得分'] = scores.round(4)
eval_df['风险排名']   = eval_df['TOPSIS得分'].rank(ascending=False).astype(int)
eval_df['风险等级']   = pd.cut(
    eval_df['TOPSIS得分'],
    bins=[0, 0.25, 0.45, 0.65, 0.80, 1.0],
    labels=['低','较低','中','较高','高']
)

eval_df = eval_df.sort_values('风险排名').reset_index(drop=True)
print(f"\n  行政区风险排名（熵权-TOPSIS）:")
print(eval_df[['行政区','TOPSIS得分','风险排名','风险等级']].to_string(index=False))

# 保存
out_csv = os.path.join(OUTPUT_DIR, 'district_topsis_ranking.csv')
eval_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

# TOPSIS排名图
fig, ax = plt.subplots(figsize=(10, 7))
color_map = {'高':'#8B0000','较高':'#E74C3C','中':'#F39C12','较低':'#A8E063','低':'#2ECC71'}
bar_colors = [color_map.get(str(v),'#888888') for v in eval_df['风险等级']]
bars = ax.barh(eval_df['行政区'], eval_df['TOPSIS得分'],
               color=bar_colors, edgecolor='white', linewidth=0.5)
for bar, val, rank in zip(bars, eval_df['TOPSIS得分'], eval_df['风险排名']):
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
            f'#{rank}  {val:.3f}', va='center', fontsize=9)
ax.set_xlabel('TOPSIS综合得分（越高风险越大）', fontsize=12)
ax.set_title('北京市各行政区城市内涝风险综合评价排名\n（熵权-TOPSIS，2018-2024年真实历史数据）', fontsize=13)
ax.axvline(0.45, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax.text(0.46, ax.get_ylim()[0]+0.1, '中等风险线', fontsize=8, color='gray')
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, min(eval_df['TOPSIS得分'].max()*1.25, 1.0))
plt.tight_layout()
out_topsis = os.path.join(OUTPUT_DIR, 'topsis_ranking.png')
plt.savefig(out_topsis, dpi=200, bbox_inches='tight')
plt.close()

print(f"\n{'='*55}")
print(f"✅ Step6完成")
print(f"  {out_risk}")
print(f"  {out_topsis}")
print(f"  {out_csv}")