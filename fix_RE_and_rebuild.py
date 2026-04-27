"""
修复RE计算并重新构造数据集
===========================
问题：原RE用6小时降雨持续时间，Ks×6h远大于日降雨，导致RE几乎全为0
修复：改用实际物理意义更准确的方案

北京短历时暴雨特征：强降雨往往集中在1~3小时内
修复方案：将日降雨量除以一个等效降雨历时系数，
          转换为等效小时雨强，再与Ks比较

新公式：
  小时雨强 I = rainfall_mm / 6    (等效6小时分布)
  入渗率   f = Ks × (1 + Psi × deficit / cum_infil)
  简化为：  f_eff = Ks × deficit_factor（mm/h）
  超渗     RE = max((I - f_eff) × 6, 0)  单位：mm

等价于：RE = max(rainfall - Ks × 6 × deficit_factor, 0)
这与原公式相同！问题在于Ks本身偏大。

真正的修复：Ks是饱和导水率，Green-Ampt里用的是初始入渗率
  初始入渗率 f0 = Ks × (1 + Psi × deficit / 初始入渗量)
  初始入渗量 F0 通常取 1mm（刚开始降雨时）
  所以 f0 = Ks × (1 + |Psi| × deficit / 1) 这比Ks大得多

更合理的简化：直接用径流曲线数（SCS-CN）的思路
  对于城市不透水面（Ks=0.01）：RE ≈ rainfall（几乎全部产流）
  对于透水土壤（Ks>50mm/h）：RE = max(rainfall - Ia, 0)
  其中Ia = 0.2 × S，S与土壤亏缺相关

最终修复方案：混合方法
  RE = rainfall × (1 - perv_ratio) + max(rainfall × perv_ratio - Ks × deficit_factor × T, 0)
  其中 perv_ratio = min(Ks/140, 1)  # Ks=140为最大值，归一化为透水率
  T = 1小时（北京短历时暴雨）
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ==================== 路径配置 ====================
LABEL_CSV = r'F:\Data\src\files\flood_labels_clean.csv'
RAIN_CSV = r'.\Step1\rainfall_daily_2018_2024.csv'
DEM_CSV = r'.\Step2\dem_soil_features.csv'
SM_CSV = r'.\Step3\soil_moisture_daily_2018_2024.csv'
OUTPUT_DIR = r'.\Step4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 读取数据 ====================
print("读取数据...")
labels = pd.read_csv(LABEL_CSV)
rain = pd.read_csv(RAIN_CSV)
dem_f = pd.read_csv(DEM_CSV)
sm_f = pd.read_csv(SM_CSV)

for df in [labels, rain, dem_f, sm_f]:
    df['lat_r'] = df['latitude'].round(6)
    df['lon_r'] = df['longitude'].round(6)

labels['date'] = labels['date'].astype(str)
rain['date'] = rain['date'].astype(str)
sm_f['date'] = sm_f['date'].astype(str)

# ==================== 修复后的RE计算函数 ====================
KS_MAX = 140.0  # 北京土壤Ks最大值（mm/h），用于归一化


def calc_RE_fixed(row):
    """
    修复版RE计算：混合城市径流模型

    物理含义：
    - 城市硬化面（Ks≈0）：降雨几乎全部形成地表径流
    - 透水土壤（Ks大）：Green-Ampt简化，降雨持续1小时
    - perv_ratio：透水面比例（由Ks归一化得到）

    公式：RE = 不透水产流 + 透水面超渗产流
    """
    rain_v = float(row.get('rainfall_mm', 0) or 0)
    ks = float(row.get('ks_mmh', 10.0) or 10.0)
    deficit = float(row.get('soil_deficit', 0.2) or 0.2)
    psi = abs(float(row.get('psi_cm', 15.0) or 15.0))

    if rain_v <= 0:
        return 0.0

    # 透水面比例（Ks越大越透水）
    perv_ratio = min(ks / KS_MAX, 1.0)
    imperv_ratio = 1.0 - perv_ratio

    # 不透水面：降雨直接产流（扣除初损2mm）
    imperv_runoff = max(0.0, rain_v - 2.0) * imperv_ratio

    # 透水面：Green-Ampt（1小时等效）
    # 考虑前期土壤湿润程度修正入渗能力
    # deficit越小（土壤越湿），入渗能力越弱
    wetness_factor = max(0.3, deficit / 0.3)  # 土壤越湿factor越小
    f_eff = ks * wetness_factor  # 有效入渗速率（mm/h）
    # 1小时内可入渗量
    infil_capacity = f_eff * 1.0  # mm
    perv_runoff = max(0.0, rain_v - infil_capacity) * perv_ratio

    RE = imperv_runoff + perv_runoff
    return round(RE, 3)


# ==================== 验证新RE公式 ====================
print("\n验证新RE公式（用正样本数据）:")
test_cases = [
    # (rainfall, ks, deficit, psi, 场景描述)
    (10, 0.01, 0.36, 10, "小雨+完全硬化路面"),
    (50, 0.01, 0.36, 10, "大雨+完全硬化路面"),
    (50, 125.0, 0.19, 8, "大雨+透水土壤(典型积水点)"),
    (50, 56.0, 0.16, 17, "大雨+中等土壤"),
    (100, 30.0, 0.25, 25, "暴雨+郊区土壤"),
    (157, 125.0, 0.15, 8, "极端暴雨+透水土壤"),
]
print(f"  {'场景':<24} {'雨量':>6} {'Ks':>6} {'亏缺':>6} {'RE':>8} {'RE/雨量':>8}")
print(f"  {'-' * 62}")
for rain_v, ks, def_, psi, desc in test_cases:
    row = {'rainfall_mm': rain_v, 'ks_mmh': ks, 'soil_deficit': def_, 'psi_cm': -psi}
    re = calc_RE_fixed(row)
    print(f"  {desc:<24} {rain_v:>6.0f} {ks:>6.1f} {def_:>6.3f} "
          f"{re:>8.2f} {re / rain_v * 100:>7.1f}%")

# ==================== 构造正样本 ====================
print("\n[1/3] 构造正样本...")
pos = labels[['date', 'lat_r', 'lon_r', 'latitude', 'longitude', 'year']].copy()
pos['label'] = 1

pos = pos.merge(rain[['date', 'lat_r', 'lon_r', 'rainfall_mm']],
                on=['date', 'lat_r', 'lon_r'], how='left')
pos = pos.merge(dem_f[['lat_r', 'lon_r', 'dem_m', 'slope_deg', 'TWI', 'HAND_m',
                       'ks_mmh', 'psi_cm', 'theta_s']],
                on=['lat_r', 'lon_r'], how='left')
pos = pos.merge(sm_f[['date', 'lat_r', 'lon_r', 'SM_prev', 'soil_deficit']],
                on=['date', 'lat_r', 'lon_r'], how='left')

pos['RE_mm'] = pos.apply(calc_RE_fixed, axis=1)
re_pos = (pos['RE_mm'] > 0).mean() * 100
print(f"  正样本RE: 均值={pos['RE_mm'].mean():.2f}mm  "
      f"最大={pos['RE_mm'].max():.2f}mm  RE>0占比={re_pos:.1f}%")

# ==================== 构造负样本 ====================
print("\n[2/3] 构造负样本...")
rain_ok = rain[rain['data_flag'] == 'OK'].copy() \
    if 'data_flag' in rain.columns else rain.copy()

pos_keys = set(zip(pos['date'], pos['lat_r'], pos['lon_r']))
pos_sites = pos[['lat_r', 'lon_r', 'latitude', 'longitude']].drop_duplicates()
flood_dates_per_site = pos.groupby(['lat_r', 'lon_r'])['date'].apply(set).to_dict()

neg_time_list = []
for _, site in pos_sites.iterrows():
    lr, lo = site['lat_r'], site['lon_r']
    flood_days = flood_dates_per_site.get((lr, lo), set())
    site_rain = rain_ok[
        (rain_ok['lat_r'] == lr) & (rain_ok['lon_r'] == lo) &
        (rain_ok['rainfall_mm'] < 5.0) & (~rain_ok['date'].isin(flood_days))
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

high_sites = dem_f[dem_f['HAND_m'] > 10][
    ['lat_r', 'lon_r', 'latitude', 'longitude']].drop_duplicates()
if len(high_sites) == 0:
    high_sites = dem_f[dem_f['slope_deg'] > 3][
        ['lat_r', 'lon_r', 'latitude', 'longitude']].drop_duplicates()

flood_dates_all = pos[['date', 'year']].drop_duplicates()
neg_space_list = []
for _, fd in flood_dates_all.iterrows():
    if len(high_sites) == 0: break
    sampled_pts = high_sites.sample(n=min(2, len(high_sites)), random_state=42)
    for _, pt in sampled_pts.iterrows():
        if (fd['date'], pt['lat_r'], pt['lon_r']) in pos_keys: continue
        r_row = rain_ok[(rain_ok['date'] == fd['date']) &
                        (rain_ok['lat_r'] == pt['lat_r']) &
                        (rain_ok['lon_r'] == pt['lon_r'])]
        rain_val = float(r_row['rainfall_mm'].values[0]) if len(r_row) > 0 else 0.0
        neg_space_list.append({
            'date': fd['date'], 'lat_r': pt['lat_r'], 'lon_r': pt['lon_r'],
            'latitude': pt['latitude'], 'longitude': pt['longitude'],
            'year': int(fd['year']), 'rainfall_mm': rain_val, 'label': 0
        })

neg_time = pd.DataFrame(neg_time_list)
neg_space = pd.DataFrame(neg_space_list) if neg_space_list else pd.DataFrame()
neg_all = pd.concat([neg_time, neg_space], ignore_index=True)
neg_all = neg_all[
    ~neg_all.apply(lambda r: (r['date'], r['lat_r'], r['lon_r']) in pos_keys, axis=1)
].drop_duplicates(subset=['date', 'lat_r', 'lon_r'])

neg_all = neg_all.merge(dem_f[['lat_r', 'lon_r', 'dem_m', 'slope_deg', 'TWI', 'HAND_m',
                               'ks_mmh', 'psi_cm', 'theta_s']],
                        on=['lat_r', 'lon_r'], how='left')
sm_med = sm_f['SM_prev'].median()
def_med = sm_f['soil_deficit'].median()
neg_all = neg_all.merge(sm_f[['date', 'lat_r', 'lon_r', 'SM_prev', 'soil_deficit']],
                        on=['date', 'lat_r', 'lon_r'], how='left')
neg_all['SM_prev'] = neg_all['SM_prev'].fillna(sm_med)
neg_all['soil_deficit'] = neg_all['soil_deficit'].fillna(def_med)
neg_all['RE_mm'] = neg_all.apply(calc_RE_fixed, axis=1)

print(f"  负样本: {len(neg_all)}条")
print(f"  负样本RE: 均值={neg_all['RE_mm'].mean():.2f}mm  "
      f"RE>0占比={(neg_all['RE_mm'] > 0).mean() * 100:.1f}%")

# ==================== 合并保存 ====================
print("\n[3/3] 合并保存...")
COLS = ['date', 'latitude', 'longitude', 'year',
        'rainfall_mm', 'RE_mm', 'SM_prev', 'soil_deficit',
        'dem_m', 'slope_deg', 'TWI', 'HAND_m',
        'ks_mmh', 'psi_cm', 'theta_s', 'label']

dataset = pd.concat([pos[COLS], neg_all[COLS]], ignore_index=True)
dataset['rainfall_mm'] = dataset['rainfall_mm'].fillna(0)
dataset['RE_mm'] = dataset['RE_mm'].fillna(0)
dataset['SM_prev'] = dataset['SM_prev'].fillna(sm_med)
dataset['soil_deficit'] = dataset['soil_deficit'].fillna(def_med)
dataset = dataset.dropna(subset=['dem_m', 'slope_deg', 'TWI', 'HAND_m'])
dataset = dataset.sort_values(['year', 'date', 'label']).reset_index(drop=True)

n_pos = (dataset['label'] == 1).sum()
n_neg = (dataset['label'] == 0).sum()

print(f"\n{'=' * 55}")
print(f"✅ 数据集构造完成")
print(f"{'=' * 55}")
print(f"  总样本: {len(dataset)}  正:{n_pos}  负:{n_neg}  比例1:{n_neg / n_pos:.1f}")

print(f"\n特征统计对比（正样本 vs 负样本）:")
feat_cols = ['rainfall_mm', 'RE_mm', 'SM_prev', 'soil_deficit', 'HAND_m', 'TWI']
for feat in feat_cols:
    pos_mean = dataset[dataset['label'] == 1][feat].mean()
    neg_mean = dataset[dataset['label'] == 0][feat].mean()
    print(f"  {feat:<14}: 正={pos_mean:>8.3f}  负={neg_mean:>8.3f}  "
          f"差异={pos_mean - neg_mean:>+8.3f}")

print(f"\n  RE>0占比: 正样本={(dataset[dataset['label'] == 1]['RE_mm'] > 0).mean() * 100:.1f}%  "
      f"负样本={(dataset[dataset['label'] == 0]['RE_mm'] > 0).mean() * 100:.1f}%")

print(f"\n时序划分:")
for name, yrs in [('训练集', list(range(2018, 2023))), ('验证集', [2023]), ('测试集', [2024])]:
    mask = dataset['year'].isin(yrs)
    n = mask.sum();
    p = dataset[mask]['label'].sum()
    print(f"  {name}({yrs[0]}~{yrs[-1] if len(yrs) > 1 else yrs[0]}): {n}条（正{p}/负{n - p}）")

out_path = os.path.join(OUTPUT_DIR, 'All_Years_Dataset.csv')
dataset.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 已保存: {out_path}")