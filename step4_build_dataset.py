"""
Step 4: 构造建模数据集（v2 - 修复 RE 计算）
============================================
修复点（来自 fix_RE_and_rebuild.py）：
  原始 RE 公式将降雨持续时间设为 6 小时，导致 Ks×6h 远大于日降雨量，
  RE 几乎全为 0，失去物理区分度。

  修复方案：混合城市径流模型
    perv_ratio = min(Ks / 140, 1)          透水面比例
    不透水面产流 = max(rainfall - 2mm, 0) × (1 - perv_ratio)
    透水面产流   = max(rainfall - Ks × wetness_factor × 1h, 0) × perv_ratio
    RE = 两者之和

  其中 wetness_factor = max(0.3, deficit / 0.3)，土壤越湿入渗能力越弱。
  北京短历时暴雨等效降雨历时取 1 小时，更符合实际。

  注：此版本同时过滤了 data_flag='MISSING' 的降雨记录，避免缺失数据污染训练集。
"""

import numpy as np
import pandas as pd
import os
np.random.seed(42)

# ==================== 路径配置 ====================
LABEL_CSV  = r'F:\Data\src\files\flood_labels_clean.csv'
RAIN_CSV   = r'.\Step1\rainfall_daily_2018_2024.csv'
DEM_CSV    = r'.\Step2\dem_soil_features.csv'
SM_CSV     = r'.\Step3\soil_moisture_daily_2018_2024.csv'
OUTPUT_DIR = r'.\Step4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== [1] 读取数据 ====================
print("读取数据...")
labels = pd.read_csv(LABEL_CSV)
rain   = pd.read_csv(RAIN_CSV)
dem_f  = pd.read_csv(DEM_CSV)
sm_f   = pd.read_csv(SM_CSV)

print(f"  标签    : {len(labels)}条")
print(f"  降雨    : {len(rain):,}条")
print(f"  地形土壤: {len(dem_f)}个点位")
print(f"  土壤水分: {len(sm_f)}条")

# 坐标精度统一
for df in [labels, rain, dem_f, sm_f]:
    df['lat_r'] = df['latitude'].round(6)
    df['lon_r'] = df['longitude'].round(6)

labels['date'] = labels['date'].astype(str)
rain['date']   = rain['date'].astype(str)
sm_f['date']   = sm_f['date'].astype(str)

# ==================== [2] 构造正样本 ====================
print("\n[1/3] 构造正样本...")
pos = labels[['date','lat_r','lon_r','latitude','longitude','year']].copy()
pos['label'] = 1

# 合并降雨
pos = pos.merge(
    rain[['date','lat_r','lon_r','rainfall_mm']],
    on=['date','lat_r','lon_r'], how='left'
)
# 合并地形
pos = pos.merge(
    dem_f[['lat_r','lon_r','dem_m','slope_deg','TWI','HAND_m',
           'ks_mmh','psi_cm','theta_s']],
    on=['lat_r','lon_r'], how='left'
)
# 合并土壤水分
pos = pos.merge(
    sm_f[['date','lat_r','lon_r','SM_prev','soil_deficit']],
    on=['date','lat_r','lon_r'], how='left'
)

print(f"  正样本: {len(pos)}条")
print(f"  缺失检查 — 降雨:{pos['rainfall_mm'].isna().sum()} "
      f"地形:{pos['dem_m'].isna().sum()} "
      f"SM:{pos['SM_prev'].isna().sum()}")

# ==================== [3] 计算超渗雨量 RE（修复版混合城市径流模型） ====================
KS_MAX = 140.0   # 北京土壤 Ks 最大值（mm/h），用于归一化透水面比例


def calc_RE(row):
    """
    混合城市径流模型（替代原 Green-Ampt 6小时版本）

    - perv_ratio：透水面比例（由 Ks/140 归一化）
    - 不透水面：扣除初损 2mm 后全部产流
    - 透水面：Green-Ampt 1 小时等效，考虑前期土壤湿润程度修正
    """
    rain_v  = float(row.get('rainfall_mm', 0) or 0)
    ks      = float(row.get('ks_mmh', 10.0)  or 10.0)
    deficit = float(row.get('soil_deficit', 0.2) or 0.2)

    if rain_v <= 0:
        return 0.0

    perv_ratio   = min(ks / KS_MAX, 1.0)
    imperv_ratio = 1.0 - perv_ratio

    # 不透水面产流（初损 2mm）
    imperv_runoff = max(0.0, rain_v - 2.0) * imperv_ratio

    # 透水面超渗产流（1 小时等效，土壤湿润修正）
    wetness_factor = max(0.3, deficit / 0.3)
    f_eff          = ks * wetness_factor          # 有效入渗速率（mm/h）
    perv_runoff    = max(0.0, rain_v - f_eff) * perv_ratio

    return round(imperv_runoff + perv_runoff, 3)

# 过滤降雨缺失记录（data_flag='MISSING' 的行不进入训练集）
before_filter = len(pos)
if 'data_flag' in pos.columns:
    pos = pos[pos['data_flag'] != 'MISSING'].reset_index(drop=True)
    dropped = before_filter - len(pos)
    if dropped > 0:
        print(f"  过滤 MISSING 降雨记录: {dropped} 条")

pos['RE_mm'] = pos.apply(calc_RE, axis=1)
re_pos_ratio = (pos['RE_mm'] > 0).mean() * 100
print(f"  RE均值={pos['RE_mm'].mean():.2f}mm  "
      f"最大={pos['RE_mm'].max():.2f}mm  "
      f"RE>0占比={re_pos_ratio:.1f}%")

# ==================== [4] 构造负样本 ====================
print("\n[2/3] 构造负样本...")

rain_ok = rain[rain.get('data_flag', 'OK') == 'OK'].copy() \
    if 'data_flag' in rain.columns else rain.copy()

pos_keys = set(zip(pos['date'], pos['lat_r'], pos['lon_r']))
pos_sites = pos[['lat_r','lon_r','latitude','longitude']].drop_duplicates()
flood_dates_per_site = pos.groupby(['lat_r','lon_r'])['date'].apply(set).to_dict()

# A) 时间负样本
print("  A) 时间负样本（同点位低雨日）...")
neg_time_list = []
for _, site in pos_sites.iterrows():
    lr, lo = site['lat_r'], site['lon_r']
    flood_days = flood_dates_per_site.get((lr, lo), set())
    site_rain = rain_ok[
        (rain_ok['lat_r'] == lr) &
        (rain_ok['lon_r'] == lo) &
        (rain_ok['rainfall_mm'] < 5.0) &
        (~rain_ok['date'].isin(flood_days))
    ]
    if len(site_rain) == 0:
        continue
    sampled = site_rain.sample(n=min(3, len(site_rain)), random_state=42)
    for _, r in sampled.iterrows():
        neg_time_list.append({
            'date': r['date'], 'lat_r': lr, 'lon_r': lo,
            'latitude': site['latitude'], 'longitude': site['longitude'],
            'year': int(r['year']), 'rainfall_mm': r['rainfall_mm'], 'label': 0
        })

neg_time = pd.DataFrame(neg_time_list)
print(f"     时间负样本: {len(neg_time)}条")

# B) 空间负样本
print("  B) 空间负样本（高HAND点位）...")
high_sites = dem_f[dem_f['HAND_m'] > 10][
    ['lat_r','lon_r','latitude','longitude']].drop_duplicates()
if len(high_sites) == 0:
    high_sites = dem_f[dem_f['slope_deg'] > 3][
        ['lat_r','lon_r','latitude','longitude']].drop_duplicates()
print(f"     高地点位（HAND>10m）: {len(high_sites)}个")

flood_dates_all = pos[['date','year']].drop_duplicates()
neg_space_list  = []
for _, fd in flood_dates_all.iterrows():
    if len(high_sites) == 0:
        break
    sampled_pts = high_sites.sample(n=min(2, len(high_sites)), random_state=42)
    for _, pt in sampled_pts.iterrows():
        key = (fd['date'], pt['lat_r'], pt['lon_r'])
        if key in pos_keys:
            continue
        r_row = rain_ok[
            (rain_ok['date'] == fd['date']) &
            (rain_ok['lat_r'] == pt['lat_r']) &
            (rain_ok['lon_r'] == pt['lon_r'])
        ]
        rain_val = float(r_row['rainfall_mm'].values[0]) \
            if len(r_row) > 0 else 0.0
        neg_space_list.append({
            'date': fd['date'], 'lat_r': pt['lat_r'], 'lon_r': pt['lon_r'],
            'latitude': pt['latitude'], 'longitude': pt['longitude'],
            'year': int(fd['year']), 'rainfall_mm': rain_val, 'label': 0
        })

neg_space = pd.DataFrame(neg_space_list) if neg_space_list else pd.DataFrame()
print(f"     空间负样本: {len(neg_space)}条")

# 合并负样本
neg_all = pd.concat([neg_time, neg_space], ignore_index=True)
neg_all = neg_all[
    ~neg_all.apply(
        lambda r: (r['date'], r['lat_r'], r['lon_r']) in pos_keys, axis=1
    )
].drop_duplicates(subset=['date','lat_r','lon_r'])

# 为负样本合并特征
neg_all = neg_all.merge(
    dem_f[['lat_r','lon_r','dem_m','slope_deg','TWI','HAND_m',
           'ks_mmh','psi_cm','theta_s']],
    on=['lat_r','lon_r'], how='left'
)
neg_all = neg_all.merge(
    sm_f[['date','lat_r','lon_r','SM_prev','soil_deficit']],
    on=['date','lat_r','lon_r'], how='left'
)
sm_med = sm_f['SM_prev'].median()
def_med= sm_f['soil_deficit'].median()
neg_all['SM_prev']      = neg_all['SM_prev'].fillna(sm_med)
neg_all['soil_deficit'] = neg_all['soil_deficit'].fillna(def_med)
neg_all['RE_mm'] = neg_all.apply(calc_RE, axis=1)

print(f"  负样本合计（去重去重叠后）: {len(neg_all)}条")

# ==================== [5] 合并保存 ====================
print("\n[3/3] 合并正负样本...")

COLS = ['date','latitude','longitude','year',
        'rainfall_mm','RE_mm',
        'SM_prev','soil_deficit',
        'dem_m','slope_deg','TWI','HAND_m',
        'ks_mmh','psi_cm','theta_s',
        'label']

dataset = pd.concat([pos[COLS], neg_all[COLS]], ignore_index=True)

# 缺失值处理
dataset['rainfall_mm']  = dataset['rainfall_mm'].fillna(0)
dataset['RE_mm']        = dataset['RE_mm'].fillna(0)
dataset['SM_prev']      = dataset['SM_prev'].fillna(sm_med)
dataset['soil_deficit'] = dataset['soil_deficit'].fillna(def_med)

before = len(dataset)
dataset = dataset.dropna(
    subset=['dem_m','slope_deg','TWI','HAND_m']
).reset_index(drop=True)
if before > len(dataset):
    print(f"  ⚠️ 删除地形缺失行: {before-len(dataset)}条")

dataset = dataset.sort_values(['year','date','label']).reset_index(drop=True)

# ==================== 输出统计 ====================
n_pos = (dataset['label']==1).sum()
n_neg = (dataset['label']==0).sum()

print(f"\n{'='*55}")
print(f"✅ 建模数据集构造完��")
print(f"{'='*55}")
print(f"  总样本量  : {len(dataset)}")
print(f"  正样本(1) : {n_pos}  负样本(0): {n_neg}  比例 1:{n_neg/n_pos:.1f}")
print(f"  年份跨度  : {dataset['year'].min()}~{dataset['year'].max()}")

print(f"\n特征统计（正样本）:")
feat_cols = ['rainfall_mm','RE_mm','SM_prev','soil_deficit',
             'dem_m','slope_deg','TWI','HAND_m']
print(dataset[dataset['label']==1][feat_cols].describe().round(3).to_string())

print(f"\n各年份分布:")
yr = dataset.groupby('year').agg(
    总数=('label','count'),
    正样本=('label','sum')
).reset_index()
yr['负样本'] = yr['总数'] - yr['正样本']
print(yr.to_string(index=False))

print(f"\n时序划分（严格按年份，禁止随机shuffle）:")
splits = [
    ('训练集', list(range(2018, 2023))),
    ('验证集', [2023]),
    ('测试集', [2024]),
]
for name, yrs in splits:
    mask = dataset['year'].isin(yrs)
    n = mask.sum()
    p = dataset[mask]['label'].sum()
    yr_str = f"{yrs[0]}" if len(yrs)==1 else f"{yrs[0]}~{yrs[-1]}"
    print(f"  {name}({yr_str}): {n}条（正{p}/负{n-p}）")

out_path = os.path.join(OUTPUT_DIR, 'All_Years_Dataset.csv')
dataset.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 已保存: {out_path}")
