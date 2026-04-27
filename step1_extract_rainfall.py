"""
Step 1: 降雨特征提取 v2
========================
改进点：
  1. 波段读取异常 → 跳过该波段，记录到错误报告
  2. 断点续传 → 已处理的年份自动跳过，中断后重新运行从断点继续
  3. 增量保存 → 每年处理完立即保存，不等全部完成
  4. 实时进度 → 每10个波段打印一次，显示当前降雨均值
  5. 生成数据质量报告 → Step1/data_report.txt
"""

import numpy as np
import pandas as pd
import rasterio
import os, json, traceback
from datetime import date, timedelta

# ==================== 路径配置 ====================
RAIN_DIR   = r'F:\Data\src\rain'
LABEL_CSV  = r'F:\Data\src\files\flood_labels_clean.csv'
OUTPUT_DIR = r'.\Step1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = list(range(2018, 2025))

# ==================== 断点续传：检查已完成年份 ====================
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'done_years': [], 'error_bands': {}}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(cp, f, indent=2)

cp = load_checkpoint()
done_years = cp.get('done_years', [])
error_bands = cp.get('error_bands', {})

if done_years:
    print(f"[断点续传] 已完成年份: {done_years}，跳过")

# ==================== 读取点位 ====================
labels = pd.read_csv(LABEL_CSV)
sites  = labels[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
print(f"积水点位数: {len(sites)}")

def get_flood_season_dates(year):
    start = date(year, 6, 1)
    return [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(107)]

# ==================== 逐年提取 ====================
for year in YEARS:
    if year in done_years:
        print(f"\n{year}年 已完成，跳过")
        continue

    tif_path = os.path.join(RAIN_DIR, f'{year}.tif')
    if not os.path.exists(tif_path):
        print(f"\n⚠️  {year}.tif 不存在，跳过")
        continue

    print(f"\n{'='*50}")
    print(f"处理 {year} 年: {tif_path}")

    dates_in_year = get_flood_season_dates(year)
    year_records  = []
    year_errors   = []      # 记录该年出错的波段

    with rasterio.open(tif_path) as src:
        n_bands   = src.count
        transform = src.transform
        height, width = src.height, src.width
        print(f"  波段数={n_bands}, 尺寸={height}×{width}")

        # 预计算行列号
        site_pixels = []
        for _, row in sites.iterrows():
            from rasterio.transform import rowcol
            r, c = rowcol(transform, row['longitude'], row['latitude'])
            r = max(0, min(int(r), height-1))
            c = max(0, min(int(c), width-1))
            site_pixels.append((r, c))
        valid_sites = list(zip(sites['latitude'], sites['longitude']))

        ok_bands   = 0
        skip_bands = 0

        for band_idx in range(min(n_bands, 107)):
            day_date  = dates_in_year[band_idx]
            band_num  = band_idx + 1

            # ==================== 核心：异常捕获 ====================
            try:
                band_data = src.read(band_num)

            except Exception as e:
                # 波段读取失败：跳过该天，填0，记录错误
                skip_bands += 1
                err_msg = f"{year}-Band{band_num}({day_date}): {type(e).__name__}: {str(e)[:80]}"
                year_errors.append(err_msg)
                print(f"\n  ⚠️ 跳过 Band{band_num}({day_date}): {type(e).__name__}")

                # 填充0（该天无降雨数据）
                for (lat, lon), (r, c) in zip(valid_sites, site_pixels):
                    year_records.append({
                        'date'       : day_date,
                        'latitude'   : lat,
                        'longitude'  : lon,
                        'rainfall_mm': 0.0,
                        'year'       : year,
                        'data_flag'  : 'MISSING'   # 标记缺失
                    })
                continue

            # 正常采样
            batch = []
            for (lat, lon), (r, c) in zip(valid_sites, site_pixels):
                val = float(band_data[r, c])
                if val < 0 or val > 2000:
                    val = 0.0
                batch.append({
                    'date'       : day_date,
                    'latitude'   : lat,
                    'longitude'  : lon,
                    'rainfall_mm': round(val, 2),
                    'year'       : year,
                    'data_flag'  : 'OK'
                })
            year_records.extend(batch)
            ok_bands += 1

            # 进度：每10天打印一次
            if band_num % 10 == 0:
                recent = [r['rainfall_mm'] for r in batch]
                print(f"  Band{band_num:3d}/107 ({day_date}) "
                      f"均值={np.mean(recent):.2f}mm  "
                      f"✅{ok_bands} ⚠️{skip_bands}", end='\r')

    print(f"\n  完成: ✅{ok_bands}天正常  ⚠️{skip_bands}天缺失")

    # ==================== 增量保存：每年完成立即写文件 ====================
    df_year = pd.DataFrame(year_records)
    year_out = os.path.join(OUTPUT_DIR, f'rainfall_{year}.csv')
    df_year.to_csv(year_out, index=False, encoding='utf-8-sig')
    print(f"  💾 已保存: {year_out} ({len(df_year):,}行)")

    # 更新断点
    cp['done_years'].append(year)
    cp['error_bands'][str(year)] = year_errors
    save_checkpoint(cp)

# ==================== 合并所有年份 ====================
print(f"\n{'='*50}")
print("合并所有年份CSV...")

all_dfs = []
for year in YEARS:
    f = os.path.join(OUTPUT_DIR, f'rainfall_{year}.csv')
    if os.path.exists(f):
        all_dfs.append(pd.read_csv(f))
        print(f"  读取 {year}: {len(all_dfs[-1]):,}行")

if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values(['year','date','latitude']).reset_index(drop=True)
    merged_path = os.path.join(OUTPUT_DIR, 'rainfall_daily_2018_2024.csv')
    df_all.to_csv(merged_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 合并完成: {merged_path} ({len(df_all):,}行)")
else:
    print("  ⚠️ 没有可合并的年份文件")
    df_all = pd.DataFrame()

# ==================== 生成数据质量报告 ====================
report_path = os.path.join(OUTPUT_DIR, 'data_report.txt')

with open(report_path, 'w', encoding='utf-8') as rpt:
    rpt.write("Step1 降雨数据质量报告\n")
    rpt.write("=" * 60 + "\n\n")

    if len(df_all) > 0:
        rpt.write(f"总记录数    : {len(df_all):,}\n")
        rpt.write(f"年份范围    : {df_all['year'].min()}~{df_all['year'].max()}\n")
        rpt.write(f"日期范围    : {df_all['date'].min()} ~ {df_all['date'].max()}\n")
        rpt.write(f"点位数      : "
                  f"{df_all[['latitude','longitude']].drop_duplicates().shape[0]}\n")

        ok_n    = (df_all['data_flag']=='OK').sum()
        miss_n  = (df_all['data_flag']=='MISSING').sum()
        rpt.write(f"\n数据完整性:\n")
        rpt.write(f"  正常记录  : {ok_n:,} ({ok_n/len(df_all)*100:.1f}%)\n")
        rpt.write(f"  缺失记录  : {miss_n:,} ({miss_n/len(df_all)*100:.1f}%)\n")

        rpt.write(f"\n降雨统计（正常数据）:\n")
        ok_rain = df_all[df_all['data_flag']=='OK']['rainfall_mm']
        rpt.write(f"  均值    : {ok_rain.mean():.3f} mm\n")
        rpt.write(f"  最大值  : {ok_rain.max():.1f} mm\n")
        rpt.write(f"  零雨天  : {(ok_rain==0).mean()*100:.1f}%\n")

        rpt.write(f"\n各年份统计:\n")
        for yr, grp in df_all.groupby('year'):
            ok = (grp['data_flag']=='OK').sum()
            ms = (grp['data_flag']=='MISSING').sum()
            mean_r = grp[grp['data_flag']=='OK']['rainfall_mm'].mean()
            max_r  = grp[grp['data_flag']=='OK']['rainfall_mm'].max()
            rpt.write(f"  {yr}: 正常{ok}条 缺失{ms}条 "
                      f"均值{mean_r:.2f}mm 最大{max_r:.1f}mm\n")

    rpt.write(f"\n缺失波段明细:\n")
    all_errors = cp.get('error_bands', {})
    if any(all_errors.values()):
        for yr, errs in all_errors.items():
            if errs:
                rpt.write(f"\n  {yr}年:\n")
                for e in errs:
                    rpt.write(f"    {e}\n")
    else:
        rpt.write("  无缺失波段 ✅\n")

    rpt.write(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"\n📋 数据报告: {report_path}")

# 控制台打印摘要
if len(df_all) > 0:
    ok_n   = (df_all['data_flag']=='OK').sum()
    miss_n = (df_all['data_flag']=='MISSING').sum()
    print(f"\n{'='*50}")
    print(f"Step1 完成摘要")
    print(f"{'='*50}")
    print(f"  总记录数  : {len(df_all):,}")
    print(f"  正常数据  : {ok_n:,} ({ok_n/len(df_all)*100:.1f}%)")
    print(f"  缺失数据  : {miss_n:,} ({miss_n/len(df_all)*100:.1f}%)")
    print(f"  降雨均值  : {df_all[df_all['data_flag']=='OK']['rainfall_mm'].mean():.2f} mm")
    print(f"\n输出文件:")
    print(f"  rainfall_YYYY.csv     每年独立文件（断点续传用）")
    print(f"  rainfall_daily_2018_2024.csv  合并文件（后续步骤用）")
    print(f"  data_report.txt       数据质量报告")
    print(f"  checkpoint.json       断点记录（重新运行自动续传）")

    # 标签匹配验证
    print(f"\n标签匹配验证:")
    label_keys = set(zip(labels['date'].astype(str),
                         labels['latitude'].round(6),
                         labels['longitude'].round(6)))
    rain_keys  = set(zip(df_all['date'],
                         df_all['latitude'].round(6),
                         df_all['longitude'].round(6)))
    matched = label_keys & rain_keys
    print(f"  标签记录: {len(label_keys)}条  匹配: {len(matched)}条  "
          f"匹配率: {len(matched)/len(label_keys)*100:.1f}%")