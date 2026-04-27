"""
Step 3: 前期土壤水分提取
=========================
输入:
  F:\Data\src\Processed_Decadal_SM_UTM50N\YYYY\SM_YYYY_MM_D_Mean_30m.tif
  （每年11个旬均值文件，覆盖汛期6月1日~9月15日）

输出:
  Step3/soil_moisture_daily_2018_2024.csv
  列: date, latitude, longitude, SM_prev, theta_s, soil_deficit, year

逻辑:
  - 积水日对应的"前期土壤水分"取该日所属旬的旬均SM
  - soil_deficit = theta_s - SM_prev（土壤吸水潜力，越大越容易入渗，越小越易积水）
  - 旬划分：1~10日=上旬(1)，11~20日=中旬(2)，21~31日=下旬(3)

注意：
  - SM的坐标系需确认是否与DEM一致（均为EPSG:4326）
  - 缺失SM用全市同旬中位数填充，不影响大样本统计
"""

import numpy as np
import pandas as pd
import rasterio
import os, warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
SM_BASE   = r'F:\Data\src\Processed_Decadal_SM_UTM50N'
THS_PATH  = r'F:\Data\src\fac\Aligned_Soil_Params\THSCH_Aligned_30m.tif'
LABEL_CSV = r'F:\Data\src\files\flood_labels_clean.csv'
DEM_FEAT  = r'.\Step2\dem_soil_features.csv'    # 含theta_s列
OUTPUT_DIR = r'.\Step3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = list(range(2018, 2025))

# ==================== 工具函数 ====================
def get_dekad(month, day):
    """返回旬编号字符串: 1=上旬, 2=中旬, 3=下旬"""
    if day <= 10:   return '1'
    elif day <= 20: return '2'
    else:           return '3'

def get_sm_filepath(year, month, dekad):
    """拼接SM文件路径"""
    return os.path.join(
        SM_BASE, str(year),
        f'SM_{year}_{month:02d}_{dekad}_Mean_30m.tif'
    )

def sample_tif_at_points(tif_path, lats, lons):
    """rasterio最近邻采样"""
    if not os.path.exists(tif_path):
        return np.full(len(lats), np.nan)
    with rasterio.open(tif_path) as src:
        nodata, data = src.nodata, src.read(1)
        h, w = src.height, src.width
        vals = []
        for lat, lon in zip(lats, lons):
            r, c = src.index(lon, lat)
            r, c = max(0,min(int(r),h-1)), max(0,min(int(c),w-1))
            v = float(data[r, c])
            if nodata is not None and abs(v - nodata) < 1e-3:
                v = np.nan
            vals.append(v)
    return np.array(vals)

# ==================== 读取标签和点位 ====================
labels = pd.read_csv(LABEL_CSV)
labels['date'] = pd.to_datetime(labels['date'])

# 从Step2结果读取theta_s（各点位孔隙度）
dem_feats = pd.read_csv(DEM_FEAT)
dem_feats['lat_r'] = dem_feats['latitude'].round(6)
dem_feats['lon_r'] = dem_feats['longitude'].round(6)
ths_lookup = dict(zip(
    zip(dem_feats['lat_r'], dem_feats['lon_r']),
    dem_feats['theta_s']
))

sites = labels[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
lats  = sites['latitude'].values
lons  = sites['longitude'].values
print(f"点位数: {len(sites)}")
print(f"标签记录数: {len(labels)}")

# ==================== 逐条记录提取SM ====================
print("\n开始提取前期土壤水分...")
print("（每个积水日→找对应旬SM文件→采样）")

# 收集所有（year, month, dekad）组合，批量处理减少文件读取次数
labels['year']   = labels['date'].dt.year
labels['month']  = labels['date'].dt.month
labels['day']    = labels['date'].dt.day
labels['dekad']  = labels.apply(lambda r: get_dekad(r['month'], r['day']), axis=1)
labels['lat_r']  = labels['latitude'].round(6)
labels['lon_r']  = labels['longitude'].round(6)

# 过滤汛期外记录（防止找不到SM文件）
labels_flood = labels[
    (labels['month'] >= 6) &
    ~((labels['month'] == 9) & (labels['day'] > 15))
].copy()
print(f"汛期内记录数: {len(labels_flood)}")

results = []
missing_files = []

# 按（year, month, dekad）分组，每组只读一次TIF
groups = labels_flood.groupby(['year','month','dekad'])
total_groups = len(groups)

for gi, ((year, month, dekad), group) in enumerate(groups):
    sm_file = get_sm_filepath(year, month, dekad)
    file_ok = os.path.exists(sm_file)

    if not file_ok:
        missing_files.append(sm_file)

    # 本组涉及的点位
    pts = group[['lat_r','lon_r','latitude','longitude']].drop_duplicates()
    sm_vals = sample_tif_at_points(sm_file, pts['latitude'], pts['longitude'])

    # 计算该旬SM中位数（用于NaN填充）
    valid_sm = sm_vals[~np.isnan(sm_vals)]
    sm_median = float(np.median(valid_sm)) if len(valid_sm) > 0 else 0.18

    sm_lut = {}
    for i, (_, pt) in enumerate(pts.iterrows()):
        key = (pt['lat_r'], pt['lon_r'])
        v   = sm_vals[i]
        sm_lut[key] = v if not np.isnan(v) else sm_median

    for _, row in group.iterrows():
        key     = (row['lat_r'], row['lon_r'])
        sm_val  = sm_lut.get(key, sm_median)
        theta_s = ths_lookup.get(key, 0.45)
        deficit = max(0.0, round(theta_s - sm_val, 4))

        results.append({
            'date'        : row['date'].strftime('%Y-%m-%d'),
            'latitude'    : row['latitude'],
            'longitude'   : row['longitude'],
            'year'        : year,
            'SM_prev'     : round(sm_val, 4),
            'theta_s'     : round(theta_s, 4),
            'soil_deficit': deficit,
        })

    if (gi+1) % 10 == 0 or (gi+1) == total_groups:
        print(f"  进度: {gi+1}/{total_groups} 组", end='\r')

print(f"\n处理完成")

# ==================== 缺失文件报告 ====================
if missing_files:
    print(f"\n⚠️ 以下SM文件不存在（已用同旬中位数填充）:")
    for f in missing_files[:10]:
        print(f"   {f}")
    if len(missing_files) > 10:
        print(f"   ...共{len(missing_files)}个缺失文件")
else:
    print("✅ 所有SM文件均存在，无缺失")

# ==================== 保存 ====================
df_sm = pd.DataFrame(results)
df_sm = df_sm.sort_values(['year','date','latitude']).reset_index(drop=True)

out_path = os.path.join(OUTPUT_DIR, 'soil_moisture_daily_2018_2024.csv')
df_sm.to_csv(out_path, index=False, encoding='utf-8-sig')

# ==================== 统计输出 ====================
print(f"\n{'='*55}")
print(f"✅ Step3完成: {out_path}")
print(f"   总行数    : {len(df_sm)}")
print(f"   年份范围  : {df_sm['year'].min()}~{df_sm['year'].max()}")
print(f"\n各年份SM统计:")
yr = df_sm.groupby('year').agg(
    记录数=('SM_prev','count'),
    SM均值=('SM_prev','mean'),
    SM最小=('SM_prev','min'),
    SM最大=('SM_prev','max'),
    亏缺均值=('soil_deficit','mean')
).round(4)
print(yr.to_string())

print(f"\n整体统计:")
print(f"  SM范围   : {df_sm['SM_prev'].min():.4f} ~ "
      f"{df_sm['SM_prev'].max():.4f} m³/m³")
print(f"  SM均值   : {df_sm['SM_prev'].mean():.4f} m³/m³")
print(f"  亏缺均值 : {df_sm['soil_deficit'].mean():.4f} m³/m³")
print(f"  亏缺范围 : {df_sm['soil_deficit'].min():.4f} ~ "
      f"{df_sm['soil_deficit'].max():.4f} m³/m³")

# 合理性检查
sm_ok      = df_sm['SM_prev'].between(0.01, 0.70).all()
deficit_ok = (df_sm['soil_deficit'] >= 0).all()
print(f"\n合理性检查:")
print(f"  SM在0.01~0.70范围内: {'✅' if sm_ok else '❌'}")
print(f"  soil_deficit≥0     : {'✅' if deficit_ok else '❌'}")