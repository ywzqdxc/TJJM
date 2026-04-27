"""
Step 1: 降雨特征提取（NC原始数据版，替代损坏的TIF）
=====================================================
输入:
  F:\Data\src\rain_nc\YYYY.nc（原始逐日降雨NC文件，每年一个）

输出格式与TIF版本完全一致:
  Step1/rainfall_YYYY.csv
  列: date, latitude, longitude, rainfall_mm, year, data_flag

核心改进（相比你们的Step1_nc.py）:
  1. 只提取102个积水点位的降雨（最近邻插值），不输出全网格
  2. 列名统一为 rainfall_mm（与TIF版本和Step4一致）
  3. 支持断点续传 + 增量保存 + 数据报告
  4. 自动跳过已完成年份（与TIF版本checkpoint共用）
"""

import numpy as np
import pandas as pd
import xarray as xr
import os, json
from datetime import date, timedelta

# ==================== 路径配置 ====================
NC_DIR     = r'F:\Data\China_Pre_1901_2024_from_CRUv4.09'   # ← 修改为你们NC文件的实际目录
LABEL_CSV  = r'F:\Data\src\files\flood_labels_clean.csv'
OUTPUT_DIR = r'.\Step1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = list(range(2018, 2025))

# ==================== 断点续传（与TIF版本共用checkpoint）====================
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'done_years': [], 'error_bands': {}}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(cp, f, indent=2, ensure_ascii=False)

cp = load_checkpoint()
done_years = cp.get('done_years', [])
if done_years:
    print(f"[断点续传] 已完成年份: {done_years}，跳过")

# ==================== 读取积水点位 ====================
labels = pd.read_csv(LABEL_CSV)
sites  = labels[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
lats_pts = sites['latitude'].values
lons_pts = sites['longitude'].values
print(f"积水点位数: {len(sites)}")

def get_flood_season_dates(year):
    start = date(year, 6, 1)
    return [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(107)]

# ==================== NC时间解析 ====================
def parse_nc_dates(ds, time_name='time'):
    """解析NC文件的时间坐标，返回 pd.DatetimeIndex"""
    time_attrs  = ds[time_name].attrs
    time_units  = time_attrs.get('units', '')
    time_values = ds[time_name].values

    if 'hours since' in time_units:
        ref = pd.Timestamp(time_units.split('since')[-1].strip())
        return pd.DatetimeIndex([ref + pd.Timedelta(hours=float(t)) for t in time_values])
    elif 'days since' in time_units:
        ref = pd.Timestamp(time_units.split('since')[-1].strip())
        return pd.DatetimeIndex([ref + pd.Timedelta(days=float(t)) for t in time_values])
    else:
        try:
            return pd.to_datetime(time_values)
        except:
            return None

# ==================== 点位最近邻采样 ====================
def find_nearest_indices(nc_lats, nc_lons, pt_lats, pt_lons):
    """
    为每个积水点位找NC网格中最近的格点索引
    返回 [(lat_idx, lon_idx), ...] 长度=len(pt_lats)
    """
    indices = []
    for pt_lat, pt_lon in zip(pt_lats, pt_lons):
        lat_idx = int(np.argmin(np.abs(nc_lats - pt_lat)))
        lon_idx = int(np.argmin(np.abs(nc_lons - pt_lon)))
        indices.append((lat_idx, lon_idx))
    return indices

# ==================== 逐年提取 ====================
for year in YEARS:
    if year in done_years:
        print(f"\n{year}年 已完成，跳过")
        continue

    # 查找NC文件（支持多种命名方式）
    nc_candidates = [
        os.path.join(NC_DIR, f'{year}.nc'),
        os.path.join(NC_DIR, f'CHM_PRE_V2_daily_{year}.nc'),
        os.path.join(NC_DIR, f'rain_{year}.nc'),
        os.path.join(NC_DIR, f'prec_{year}.nc'),
    ]
    nc_path = next((p for p in nc_candidates if os.path.exists(p)), None)

    if nc_path is None:
        print(f"\n⚠️  {year}年NC文件未找到，跳过")
        print(f"   查找路径: {nc_candidates[0]} 等")
        continue

    file_mb = os.path.getsize(nc_path) / 1024**2
    print(f"\n{'='*50}")
    print(f"处理 {year} 年: {nc_path} ({file_mb:.0f} MB)")

    # 打开NC
    ds = xr.open_dataset(nc_path, decode_times=False)

    # 找变量名
    var_name = None
    for v in ['prec', 'pre', 'pr', 'rain', 'precip', 'PRE', 'precipitation']:
        if v in ds.data_vars:
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]
    print(f"  降雨变量: {var_name}")

    # 找坐标名
    lat_name = next((c for c in ['lat','latitude','Lat'] if c in ds.coords), None)
    lon_name = next((c for c in ['lon','longitude','Lon'] if c in ds.coords), None)
    time_name= next((c for c in ['time','TIME','Time'] if c in ds.coords), None)

    nc_lats = ds[lat_name].values
    nc_lons = ds[lon_name].values
    print(f"  NC分辨率: {abs(nc_lats[1]-nc_lats[0]):.4f}° ≈ "
          f"{abs(nc_lats[1]-nc_lats[0])*111:.1f}km")
    print(f"  NC范围: lat {nc_lats.min():.2f}~{nc_lats.max():.2f}, "
          f"lon {nc_lons.min():.2f}~{nc_lons.max():.2f}")

    # 解析日期
    all_dates = parse_nc_dates(ds, time_name)
    if all_dates is None:
        print(f"  ❌ 无法解析时间坐标，跳过")
        ds.close()
        continue

    # 筛选汛期（6月1日~9月15日）
    flood_start = pd.Timestamp(f'{year}-06-01')
    flood_end   = pd.Timestamp(f'{year}-09-15')
    flood_mask  = (all_dates >= flood_start) & (all_dates <= flood_end)
    flood_idx   = np.where(flood_mask)[0]
    flood_dates = all_dates[flood_mask]

    print(f"  NC总时间步: {len(all_dates)}天")
    print(f"  汛期匹配天数: {len(flood_idx)}天 "
          f"({flood_dates[0].strftime('%Y-%m-%d')} ~ "
          f"{flood_dates[-1].strftime('%Y-%m-%d')})")

    if len(flood_idx) == 0:
        print(f"  ❌ 汛期无数据，跳过")
        ds.close()
        continue

    # 预计算点位的最近邻格点索引
    pt_indices = find_nearest_indices(nc_lats, nc_lons, lats_pts, lons_pts)

    # 验证最近邻距离（确保NC覆盖北京）
    max_dist = max(
        max(abs(nc_lats[li] - lats_pts[i]), abs(nc_lons[lj] - lons_pts[i]))
        for i, (li, lj) in enumerate(pt_indices)
    )
    print(f"  最近邻最大误差: {max_dist:.4f}° ≈ {max_dist*111:.1f}km")
    if max_dist > 0.5:
        print(f"  ⚠️ 误差超过0.5°，请确认NC文件覆盖北京范围")

    # 逐天提取
    year_records = []
    ok_days = 0
    target_dates = get_flood_season_dates(year)  # 标准107天日期列表

    print(f"  开始逐天采样（共107天标准汛期）...")
    print(f"  {'天数':<6} {'日期':<12} {'均值mm':>8} {'最大mm':>8} {'状态'}")
    print(f"  {'-'*46}")

    # 建立NC日期→索引的查找表（应对NC天数可能≠107的情况）
    nc_date_lut = {d.strftime('%Y-%m-%d'): i
                   for i, d in zip(flood_idx, flood_dates)}

    for day_num, day_date_str in enumerate(target_dates):
        t_idx = nc_date_lut.get(day_date_str)

        if t_idx is None:
            # 该天NC没有数据（填0，标记MISSING）
            for (lat, lon) in zip(lats_pts, lons_pts):
                year_records.append({
                    'date': day_date_str, 'latitude': lat,
                    'longitude': lon, 'rainfall_mm': 0.0,
                    'year': year, 'data_flag': 'MISSING'
                })
            if (day_num+1) % 10 == 0:
                print(f"  {day_num+1:<6} {day_date_str:<12} {'---':>8} {'---':>8} ⚠️MISSING")
            continue

        try:
            # 读取该天的2D数据
            data_2d = ds[var_name].isel(**{time_name: int(t_idx)}).values

            # 在102个点位处采样
            vals = []
            for (lat, lon), (li, lj) in zip(
                    zip(lats_pts, lons_pts), pt_indices):
                v = float(data_2d[li, lj])
                if np.isnan(v) or v < 0 or v > 2000:
                    v = 0.0
                vals.append(round(v, 2))
                year_records.append({
                    'date': day_date_str, 'latitude': lat,
                    'longitude': lon, 'rainfall_mm': v,
                    'year': year, 'data_flag': 'OK'
                })

            ok_days += 1
            if (day_num+1) % 10 == 0 or day_num < 3:
                print(f"  {day_num+1:<6} {day_date_str:<12} "
                      f"{np.mean(vals):>8.2f} {np.max(vals):>8.2f} ✅")

        except Exception as e:
            # 读取失败填0
            for (lat, lon) in zip(lats_pts, lons_pts):
                year_records.append({
                    'date': day_date_str, 'latitude': lat,
                    'longitude': lon, 'rainfall_mm': 0.0,
                    'year': year, 'data_flag': 'MISSING'
                })
            print(f"  {day_num+1:<6} {day_date_str:<12} "
                  f"{'---':>8} {'---':>8} ⚠️{type(e).__name__}")

    ds.close()
    miss_days = 107 - ok_days
    print(f"\n  完成: ✅{ok_days}天正常  ⚠️{miss_days}天缺失")

    # 保存该年CSV
    df_year = pd.DataFrame(year_records)
    out_csv  = os.path.join(OUTPUT_DIR, f'rainfall_{year}.csv')
    df_year.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"  💾 已保存: {out_csv} ({len(df_year):,}行)")

    # 更新断点
    cp['done_years'] = cp.get('done_years', []) + [year]
    if str(year) not in cp.get('error_bands', {}):
        cp.setdefault('error_bands', {})[str(year)] = []
    save_checkpoint(cp)

# ==================== 合并 + 报告 ====================
print(f"\n{'='*50}")
print("合并所有年份...")
all_dfs = []
for year in YEARS:
    f = os.path.join(OUTPUT_DIR, f'rainfall_{year}.csv')
    if os.path.exists(f):
        df_tmp = pd.read_csv(f)
        all_dfs.append(df_tmp)
        ok  = (df_tmp['data_flag']=='OK').sum() if 'data_flag' in df_tmp.columns else len(df_tmp)
        mis = (df_tmp['data_flag']=='MISSING').sum() if 'data_flag' in df_tmp.columns else 0
        print(f"  {year}: {len(df_tmp):,}行  ✅{ok}  ⚠️{mis}")

if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values(['year','date','latitude']).reset_index(drop=True)
    merged = os.path.join(OUTPUT_DIR, 'rainfall_daily_2018_2024.csv')
    df_all.to_csv(merged, index=False, encoding='utf-8-sig')
    print(f"\n✅ 合并完成: {merged} ({len(df_all):,}行)")

    # 标签匹配验证
    labels2 = pd.read_csv(LABEL_CSV)
    label_keys = set(zip(labels2['date'].astype(str),
                         labels2['latitude'].round(6),
                         labels2['longitude'].round(6)))
    rain_keys  = set(zip(df_all['date'],
                         df_all['latitude'].round(6),
                         df_all['longitude'].round(6)))
    matched = label_keys & rain_keys
    print(f"\n标签匹配: {len(matched)}/{len(label_keys)} = "
          f"{len(matched)/len(label_keys)*100:.1f}%")
    if len(matched) < len(label_keys):
        unmatched = label_keys - rain_keys
        print(f"未匹配（前3条）:")
        for u in list(unmatched)[:3]:
            print(f"  {u}")