"""
Step 6 v3: 彻底解决全红问题
=============================
根因分析：
  训练集中正负样本的降雨差异极大（正样本均值53mm，负样本均值7mm）
  导致Logistic回归给rainfall_mm赋予极大系数（OR=60.9）
  只要输入的rainfall值在正样本范围内（>20mm），模型概率就趋近于1
  地形特征（HAND/TWI）的系数相对太小，无法在全红背景上产生差异

真正的修复方案：
  风险区划不应依赖Logistic回归的绝对概率值
  而应依赖各像元之间的"相对风险差异"

  做法：固定rainfall=0（排除降雨驱动），
  只让地形特征变化，得到"纯地形风险得分"
  再用分位数分级，而非固定0.2/0.4/0.6/0.8阈值

  物理解释：这张图回答的是"在降雨发生时，哪里更容易积水"
  而非"给定降雨量时是否会积水"
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

DATASET_CSV = r'.\Step4\All_Years_Dataset.csv'
LABEL_CSV   = r'F:\Data\src\files\flood_labels_clean.csv'
RAIN_CSV    = r'.\Step1\rainfall_daily_2018_2024.csv'
DEM_PATH    = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
DEM_FEAT    = r'.\Step2\dem_soil_features.csv'
OUTPUT_DIR  = r'.\Step6'
os.makedirs(OUTPUT_DIR, exist_ok=True)
KS_MAX = 140.0

# ==================== [1] 训练两个模型 ====================
print("[1/4] 训练模型...")
df = pd.read_csv(DATASET_CSV)
train = df[df['year'].between(2018,2023)].copy()
train['lat_bin'] = pd.cut(train['latitude'],  bins=5, labels=False)
train['lon_bin'] = pd.cut(train['longitude'], bins=5, labels=False)

# 模型A：全特征（用于TOPSIS数值计算）
FEAT_ALL = ['rainfall_mm','RE_mm','HAND_m','TWI','slope_deg','dem_m','theta_s']
feat_use = FEAT_ALL + ['lat_bin','lon_bin']
X_tr_all = train[feat_use].fillna(0).values
y_tr = train['label'].values
sc_all = StandardScaler()
X_tr_all_s = sc_all.fit_transform(X_tr_all)
m_all = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
m_all.fit(X_tr_all_s, y_tr)

# 模型B：纯地形模型（rainfall/RE置0，仅用地形特征做空间区划）
# 物理含义：控制降雨量后，哪个地方的地形条件更容易汇水积水
FEAT_TOPO = ['HAND_m','TWI','slope_deg','dem_m','theta_s']
feat_topo = FEAT_TOPO + ['lat_bin','lon_bin']
X_tr_topo = train[feat_topo].fillna(0).values
sc_topo = StandardScaler()
X_tr_topo_s = sc_topo.fit_transform(X_tr_topo)
m_topo = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
m_topo.fit(X_tr_topo_s, y_tr)

pos = train[train['label']==1]
print(f"  模型A（全特征）训练完成")
print(f"  模型B（纯地形）训练完成")
print(f"  正样本HAND中位={pos['HAND_m'].median():.1f}m  TWI中位={pos['TWI'].median():.2f}")

# ==================== [2] DEM地形特征估算 ====================
print("\n[2/4] 读取DEM并估算地形特征...")
dem_feats = pd.read_csv(DEM_FEAT)

with rasterio.open(DEM_PATH) as src:
    dem_full = src.read(1).astype(np.float32)
    height, width = dem_full.shape

STRIDE = 10
dem_ds = dem_full[::STRIDE, ::STRIDE]
H, W   = dem_ds.shape

res_m = 30.0 * STRIDE
dy = np.gradient(dem_ds, axis=0) / res_m
dx = np.gradient(dem_ds, axis=1) / res_m
slope_est = np.clip(np.rad2deg(np.arctan(np.sqrt(dx**2+dy**2))), 0, 60)

dem_valid = dem_ds.copy()
dem_valid[dem_valid < -100] = np.nan
dem_norm = (dem_valid - np.nanmin(dem_valid)) / \
           (np.nanmax(dem_valid) - np.nanmin(dem_valid) + 1e-6)

# 地形特征映射：低洼地→HAND小、TWI大（风险高）
hand_min, hand_max = dem_feats['HAND_m'].min(), dem_feats['HAND_m'].max()
twi_min,  twi_max  = dem_feats['TWI'].min(),    dem_feats['TWI'].max()
hand_est = hand_min + dem_norm * (hand_max - hand_min)
twi_est  = twi_max  - dem_norm * (twi_max  - twi_min)
twi_est  = np.clip(twi_est, twi_min, twi_max)
theta_s  = dem_feats['theta_s'].median()

nodata_mask = np.isnan(dem_ds) | (dem_ds < -100)
print(f"  HAND: {np.nanmin(hand_est):.1f}~{np.nanmax(hand_est):.1f}m  "
      f"TWI: {np.nanmin(twi_est):.2f}~{np.nanmax(twi_est):.2f}")

# ==================== [3] 纯地形风险预测 ====================
print("\n[3/4] 纯地形风险预测（控制降雨，展示空间差异）...")

feat_topo_grid = np.zeros((H*W, 7), dtype=np.float32)
feat_topo_grid[:,0] = hand_est.ravel()   # HAND_m
feat_topo_grid[:,1] = twi_est.ravel()    # TWI
feat_topo_grid[:,2] = slope_est.ravel()  # slope_deg
feat_topo_grid[:,3] = dem_ds.ravel()     # dem_m
feat_topo_grid[:,4] = theta_s            # theta_s
feat_topo_grid[:,5] = 2                  # lat_bin
feat_topo_grid[:,6] = 2                  # lon_bin

feat_topo_s = sc_topo.transform(feat_topo_grid)
proba_topo  = m_topo.predict_proba(feat_topo_s)[:,1].reshape(H, W)
proba_topo[nodata_mask] = np.nan

print(f"  地形风险概率范围: {np.nanmin(proba_topo):.3f}~{np.nanmax(proba_topo):.3f}")
print(f"  标准差: {np.nanstd(proba_topo):.4f}（>0.05说明有空间差异）")

# 用分位数分级（保证各级都有像元）
valid_p = proba_topo[~np.isnan(proba_topo)].ravel()
q20, q40, q60, q80 = np.percentile(valid_p, [20, 40, 60, 80])
print(f"  分位数阈值: 20%={q20:.3f} 40%={q40:.3f} 60%={q60:.3f} 80%={q80:.3f}")

rl_topo = np.full_like(proba_topo, np.nan)
rl_topo[proba_topo <  q20] = 1
rl_topo[(proba_topo>=q20)&(proba_topo<q40)] = 2
rl_topo[(proba_topo>=q40)&(proba_topo<q60)] = 3
rl_topo[(proba_topo>=q60)&(proba_topo<q80)] = 4
rl_topo[proba_topo >= q80] = 5
rl_topo[nodata_mask] = np.nan

level_names = ['极低','低','中','高','极高']
pcts = [np.nanmean(rl_topo==i)*100 for i in range(1,6)]
print(f"  各级比例: " + "  ".join([f"{n}:{p:.1f}%" for n,p in zip(level_names,pcts)]))

# ==================== [4] 出图 ====================
print("\n生成风险区划图...")
colors5 = ['#2ECC71','#A8E063','#F39C12','#E74C3C','#8B0000']
cmap5   = mcolors.ListedColormap(colors5)
norm5   = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], 5)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('北京市城市内涝地形风险区划（2018-2024）', fontsize=14, fontweight='bold')

# 左图：地形风险概率（连续）
ax = axes[0]
im = ax.imshow(proba_topo, cmap='RdYlGn_r', vmin=valid_p.min(),
               vmax=valid_p.max(), aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='地形积水易发性指数')
ax.set_title('地形积水易发性概率分布\n（基于纯地形特征的Logistic回归）', fontsize=11)
ax.set_xlabel('经向像元')
ax.set_ylabel('纬向像元')
ax.text(0.02, 0.02,
        '低洼地（HAND小、TWI大）→高易发性\n山区（高程高、坡度大）→低易发性',
        transform=ax.transAxes, fontsize=8, color='gray',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 右图：五级分类（分位数）
ax = axes[1]
im = ax.imshow(rl_topo, cmap=cmap5, norm=norm5, aspect='auto')
cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[1,2,3,4,5])
cb.set_ticklabels(level_names)
cb.set_label('风险等级')
stats_text = '\n'.join([f"{n}:  {p:.1f}%" for n,p in zip(level_names, pcts)])
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
ax.set_title('北京市城市内涝五级风险区划\n（分位数分级，各级约20%像元）', fontsize=11)
ax.set_xlabel('经向像元')
ax.set_ylabel('纬向像元')

plt.tight_layout()
out_map = os.path.join(OUTPUT_DIR, 'risk_map.png')
plt.savefig(out_map, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✅ {out_map}")

# ==================== [5] TOPSIS ====================
print("\n[4/4] 熵权-TOPSIS行政区评价...")
labels_df = pd.read_csv(LABEL_CSV)
rain_df   = pd.read_csv(RAIN_CSV)

districts = {
    '东城区':(39.88,39.96,116.36,116.44),'西城区':(39.87,39.96,116.30,116.42),
    '朝阳区':(39.83,40.05,116.39,116.64),'丰台区':(39.75,39.92,116.17,116.42),
    '海淀区':(39.89,40.15,116.17,116.41),'石景山区':(39.88,39.96,116.13,116.26),
    '通州区':(39.78,40.02,116.55,116.82),'顺义区':(40.00,40.22,116.44,116.74),
    '昌平区':(40.08,40.32,116.08,116.46),'房山区':(39.62,39.93,115.83,116.23),
    '密云区':(40.28,40.54,116.62,117.02),'延庆区':(40.38,40.62,115.73,116.08),
}

rows = []
for dist,(lt_mn,lt_mx,ln_mn,ln_mx) in districts.items():
    sub  = labels_df[labels_df['latitude'].between(lt_mn,lt_mx)&
                     labels_df['longitude'].between(ln_mn,ln_mx)]
    freq = round(len(sub)/labels_df['year'].nunique(),2)
    mdep = round(sub['depth_real_cm'].mean(),1) if len(sub)>0 else 0.0
    xdep = round(sub['depth_real_cm'].max(),1)  if len(sub)>0 else 0.0
    sub_r = rain_df[rain_df['latitude'].between(lt_mn,lt_mx)&
                    rain_df['longitude'].between(ln_mn,ln_mx)]
    avgr = round(sub_r['rainfall_mm'].mean(),2) if len(sub_r)>0 else 0.0
    sub_d = dem_feats[dem_feats['latitude'].between(lt_mn,lt_mx)&
                      dem_feats['longitude'].between(ln_mn,ln_mx)]
    ah  = round(sub_d['HAND_m'].mean(),2) if len(sub_d)>0 else 5.0
    aks = round(sub_d['ks_mmh'].mean(),2) if len(sub_d)>0 else 56.0
    rows.append({'行政区':dist,'年均积水频次':freq,'平均积水深cm':mdep,
                 '最大积水深cm':xdep,'汛期均降雨mm':avgr,
                 '地势低洼指数':round(1/(ah+1),4),
                 '不透水代理指数':round(max(0,1-aks/KS_MAX),4)})

eval_df = pd.DataFrame(rows)
INDS = ['年均积水频次','平均积水深cm','最大积水深cm',
        '汛期均降雨mm','地势低洼指数','不透水代理指数']
X = eval_df[INDS].values.astype(float)

def ew(X):
    s=X.sum(0)+1e-10; Xn=np.clip(X/s,1e-10,1)
    e=-np.sum(Xn*np.log(Xn),0)/np.log(len(X))
    return (1-e)/(np.sum(1-e)+1e-10)

def tp(X,w):
    n=np.sqrt((X**2).sum(0))+1e-10; Xw=(X/n)*w
    p=Xw.max(0); ng=Xw.min(0)
    return np.sqrt(((Xw-ng)**2).sum(1))/(
           np.sqrt(((Xw-p)**2).sum(1))+np.sqrt(((Xw-ng)**2).sum(1))+1e-10)

w = ew(X)
print("  权重:", {n:round(v,3) for n,v in zip(INDS,w)})
eval_df['TOPSIS得分'] = tp(X,w).round(4)
eval_df['风险排名']   = eval_df['TOPSIS得分'].rank(ascending=False).astype(int)
eval_df['风险等级']   = pd.cut(eval_df['TOPSIS得分'],
    bins=[0,.25,.45,.65,.80,1.0], labels=['低','较低','中','较高','高'])
eval_df = eval_df.sort_values('风险排名').reset_index(drop=True)
print(eval_df[['行政区','TOPSIS得分','风险排名','风险等级']].to_string(index=False))
eval_df.to_csv(os.path.join(OUTPUT_DIR,'district_topsis_ranking.csv'),
               index=False, encoding='utf-8-sig')

# 排名图
fig, ax = plt.subplots(figsize=(10,7))
cm_r = {'高':'#8B0000','较高':'#E74C3C','中':'#F39C12','较低':'#A8E063','低':'#2ECC71'}
bcs  = [cm_r.get(str(v),'#888') for v in eval_df['风险等级']]
bars = ax.barh(eval_df['行政区'], eval_df['TOPSIS得分'],
               color=bcs, edgecolor='white', linewidth=0.5)
for bar,val,rank in zip(bars,eval_df['TOPSIS得分'],eval_df['风险排名']):
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
            f'#{rank}  {val:.3f}', va='center', fontsize=9)
ax.axvline(0.45, color='gray', ls='--', lw=1, alpha=0.6)
ax.set_xlabel('TOPSIS综合得分', fontsize=12)
ax.set_title('北京市各行政区城市内涝风险综合评价\n（熵权-TOPSIS，2018-2024年真实数据）', fontsize=13)
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, eval_df['TOPSIS得分'].max()*1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'topsis_ranking.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\n{'='*55}")
print(f"✅ Step6 v3完成")
print(f"  risk_map.png           地形风险区划图")
print(f"  topsis_ranking.png     行政区排名图")
print(f"  district_topsis_ranking.csv")