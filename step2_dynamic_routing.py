"""
Step 2 (v7.0): 动态水文物理汇流模型
======================================
v7.0 相对 v6.0 的唯一改动（★ 标记处）：
  在保存多年均值之后，新增逐年降雨量和CR文件保存
  → Precipitation_2012.npy ... Precipitation_2024.npy
  → CR_2012.npy ... CR_2024.npy
  供 Step3 v7.0 逐年脆弱性计算使用

其余逻辑与 v6.0 完全一致。
"""

import numpy as np
import xarray as xr
from pysheds.grid import Grid
from scipy.ndimage import zoom
import pandas as pd
import rasterio
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
NC_DIR     = r'E:\Data\China_Pre_1901_2024_from_CRUv4.09'
SM_DIR     = r'E:\Data\src\Processed_Decadal_SM_UTM50N'
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
STATIC_DIR = r'./Step_New/Static'
OUTPUT_DIR = r'./Step_New/Dynamic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CR_URBAN = 0.90
CR_ROCK  = 0.70
YEARS    = list(range(2012, 2025))
DIRMAP   = (64, 128, 1, 2, 4, 8, 16, 32)

def get_decade(day):
    if day <= 10: return 1
    elif day <= 20: return 2
    else: return 3

print("=" * 70)
print("Step 2 (v7.0)：动态水文物理汇流模型")
print("v7.0新增：逐年 Precipitation_{year}.npy + CR_{year}.npy")
print("=" * 70)

# ============================================================
# 1. 加载静态数据
# ============================================================
print("\n[1/6] 加载静态数据...")

fdir_raw    = np.load(os.path.join(STATIC_DIR, 'fdir.npy'))
slope       = np.load(os.path.join(STATIC_DIR, 'slope.npy'))
hand        = np.load(os.path.join(STATIC_DIR, 'hand.npy'))
ks          = np.load(os.path.join(STATIC_DIR, 'ks.npy'))
ths         = np.load(os.path.join(STATIC_DIR, 'ths.npy'))
urban_mask  = np.load(os.path.join(STATIC_DIR, 'urban_mask.npy')).astype(bool)
rock_mask   = np.load(os.path.join(STATIC_DIR, 'rock_mask.npy')).astype(bool)
nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)

h, w       = fdir_raw.shape
valid_mask = ~nodata_mask
total_valid = int(valid_mask.sum())

print(f"    栅格尺寸: {h} × {w}  有效像元: {total_valid:,}")

grid         = Grid.from_raster(DEM_PATH)
base_raster  = grid.read_raster(DEM_PATH)
fdir_obj     = base_raster.copy()
fdir_obj[:]  = fdir_raw
fdir_obj.nodata = 0

ks_nat  = ks[~np.isnan(ks) & ~urban_mask & ~rock_mask]
ks_norm = np.zeros_like(ks)
if ks_nat.size > 0:
    ks_norm = (ks - ks_nat.min()) / (ks_nat.max() - ks_nat.min() + 1e-6)
ks_norm = np.where(np.isnan(ks), 0.0, np.clip(ks_norm, 0, 1))
print(f"    Ks归一化: 有效={ks_nat.size:,}  范围=[{ks_nat.min():.4f},{ks_nat.max():.4f}] mm/h")


# ============================================================
# 2. 逐年计算
# ============================================================
print("\n[2/6] 逐年计算（CHM_PRE V2，汛期6/15-9/15）...")

year_acc_raw    = {}
year_rain_grids = {}
year_cr_arrays  = {}

for year in YEARS:
    nc_path = os.path.join(NC_DIR, f'CHM_PRE_V2_daily_{year}.nc')
    if not os.path.exists(nc_path):
        nc_path = os.path.join(NC_DIR, f'{year}.nc')
    if not os.path.exists(nc_path):
        print(f"  {year}: 跳过（无降雨文件）")
        continue

    print(f"  {year}...", end=' ', flush=True)

    ds        = xr.open_dataset(nc_path, engine='netcdf4')
    var_name  = 'prec' if 'prec' in ds.data_vars else list(ds.data_vars)[0]
    rain_data = ds[var_name].sel(lat=slice(39.4, 41.1), lon=slice(115.4, 117.5))
    all_dates = pd.to_datetime(ds['time'].values)

    flood_mask = (
        ((all_dates.month > 6) | ((all_dates.month == 6) & (all_dates.day >= 15))) &
        ((all_dates.month < 9) | ((all_dates.month == 9) & (all_dates.day <= 15)))
    )
    if not flood_mask.any():
        print("无汛期数据"); ds.close(); continue

    flood_rain  = rain_data[flood_mask]
    flood_dates = all_dates[flood_mask]
    daily_mean  = flood_rain.mean(dim=['lat', 'lon']).values
    n_days      = int(flood_mask.sum())

    peak_date  = flood_dates[np.argmax(daily_mean)]
    decade_idx = get_decade(peak_date.day)
    sm_file    = os.path.join(SM_DIR, str(year),
                              f'SM_{year}_{peak_date.month:02d}_{decade_idx}_Mean_30m.tif')

    if os.path.exists(sm_file):
        with rasterio.open(sm_file) as src:
            initial_sm = src.read(1, out_shape=(h, w)).astype(np.float32)
        initial_sm = np.where(initial_sm < 0, 0.14, initial_sm)
        initial_sm = np.where(np.isnan(initial_sm), 0.25, initial_sm)  # v6.0 NaN修复
    else:
        initial_sm = np.full((h, w), 0.25, dtype=np.float32)

    season_rain_nc = flood_rain.sum(dim='time').values
    h_r, w_r = season_rain_nc.shape
    scale_y, scale_x = h / h_r, w / w_r
    if abs(scale_y - 1.0) < 1e-6 and abs(scale_x - 1.0) < 1e-6:
        rain_grid = season_rain_nc
    else:
        rain_grid = zoom(season_rain_nc, (scale_y, scale_x), order=1)
    season_rain_m = np.clip(rain_grid, 0, 3000)

    soil_deficit = np.clip(ths - initial_sm, 0.0, None)
    soil_deficit[urban_mask] = 0.0
    soil_deficit[rock_mask]  = 0.0
    infil_factor = np.clip(soil_deficit * ks_norm * 2.0, 0.0, 0.70)
    runoff_coeff = np.clip(0.9 - infil_factor, 0.05, 0.95)
    runoff_coeff[urban_mask] = CR_URBAN
    runoff_coeff[rock_mask]  = CR_ROCK

    effective_runoff  = season_rain_m * runoff_coeff
    weights_raster    = base_raster.copy()
    weights_raster[:] = effective_runoff
    weights_raster.nodata = 0

    try:
        acc_water = grid.accumulation(fdir_obj, dirmap=DIRMAP, weights=weights_raster)
        acc_arr   = np.array(acc_water, dtype=np.float32)
        year_acc_raw[year] = acc_arr

        year_rain_grids[year] = season_rain_m.astype(np.float32)
        rc_save = runoff_coeff.copy().astype(np.float32)
        rc_save[nodata_mask] = np.nan
        year_cr_arrays[year] = rc_save

        rain_v = season_rain_m[valid_mask]
        cr_v   = runoff_coeff[valid_mask & ~np.isnan(runoff_coeff)]
        print(f"✓ ({n_days}天)  雨量={rain_v.mean():.1f}mm  CR={cr_v.mean():.4f}")
    except Exception as e:
        print(f"✗ 汇流失败: {e}")

    ds.close()

print(f"\n  成功计算: {len(year_acc_raw)}/{len(YEARS)} 年")


# ============================================================
# 2.5 保存多年均值（与v6.0完全一致）
# ============================================================
print("\n[★ 2.5] 保存多年均值...")

if year_rain_grids:
    rain_stack = np.stack(list(year_rain_grids.values()), axis=0)
    rain_mean  = np.nanmean(rain_stack, axis=0).astype(np.float32)
    rain_mean  = np.where(nodata_mask, np.nan, rain_mean)
    np.save(os.path.join(OUTPUT_DIR, 'Precipitation_Mean.npy'), rain_mean)
    v_r = rain_mean[valid_mask & ~np.isnan(rain_mean)]
    print(f"  ✅ Precipitation_Mean.npy  均值={v_r.mean():.1f}mm")

if year_cr_arrays:
    cr_stack = np.stack(list(year_cr_arrays.values()), axis=0)
    cr_mean  = np.nanmean(cr_stack, axis=0).astype(np.float32)
    cr_mean  = np.where(nodata_mask, np.nan, cr_mean)
    np.save(os.path.join(OUTPUT_DIR, 'CR_MultiYear_Mean.npy'), cr_mean)
    v_c = cr_mean[valid_mask & ~np.isnan(cr_mean)]
    print(f"  ✅ CR_MultiYear_Mean.npy  均值={v_c.mean():.4f}")


# ============================================================
# ★ 2.6 新增：保存逐年降雨量和CR文件（v7.0新增）
# ============================================================
print("\n[★ 2.6] 保存逐年降雨量和CR文件（供Step3逐年脆弱性计算）...")

saved_rain = 0
saved_cr   = 0

for yr, rain_arr in year_rain_grids.items():
    rain_save = rain_arr.copy()
    rain_save[nodata_mask] = np.nan
    out_path = os.path.join(OUTPUT_DIR, f'Precipitation_{yr}.npy')
    np.save(out_path, rain_save.astype(np.float32))
    saved_rain += 1

for yr, cr_arr in year_cr_arrays.items():
    out_path = os.path.join(OUTPUT_DIR, f'CR_{yr}.npy')
    np.save(out_path, cr_arr.astype(np.float32))
    saved_cr += 1

print(f"  ✅ Precipitation_{{year}}.npy × {saved_rain} 个年份")
print(f"  ✅ CR_{{year}}.npy           × {saved_cr} 个年份")

# 验证范围
print(f"\n  逐年降雨量统计:")
for yr in sorted(year_rain_grids.keys()):
    fp = os.path.join(OUTPUT_DIR, f'Precipitation_{yr}.npy')
    if os.path.exists(fp):
        arr = np.load(fp)
        v   = arr[valid_mask & ~np.isnan(arr)]
        print(f"    {yr}: 均值={v.mean():.1f}mm  范围=[{v.min():.1f},{v.max():.1f}]")


# ============================================================
# 3-6. WDI计算（与v6.0完全一致）
# ============================================================
print("\n[3/6] 计算全局WDI缩放因子...")

_chunks     = [arr[~nodata_mask & (arr > 1e-12)] for arr in year_acc_raw.values()]
_chunks     = [c for c in _chunks if c.size > 0]
all_acc_pos = np.concatenate(_chunks) if _chunks else np.array([])

if all_acc_pos.size > 100:
    global_acc_p75 = float(np.percentile(all_acc_pos, 75))
    global_scale   = (np.exp(6.0) - 1.0) / max(global_acc_p75, 1e-12)
    print(f"    P75={global_acc_p75:.6e}  Sg={global_scale:.4e}")
else:
    global_scale = 1e8

print("\n[4/6] 计算各年WDI...")

slope_s   = np.clip(slope, 0.5, 60.0).astype(np.float64)
hand_s    = np.clip(hand,  0.0, 500.0).astype(np.float64)
log_denom = np.log1p(slope_s) * np.log1p(hand_s)

year_raw_wdi = {}
max_wdi = np.full((h, w), -np.inf, dtype=np.float32)

for year, acc_arr_y in year_acc_raw.items():
    log_acc = np.log1p(acc_arr_y.astype(np.float64) * global_scale)
    raw_wdi = np.where(
        nodata_mask | (log_denom < 0.1),
        np.nan,
        np.clip(log_acc / (log_denom + 1e-8), -100, 100)
    ).astype(np.float32)
    year_raw_wdi[year] = raw_wdi
    np.maximum(max_wdi, np.nan_to_num(raw_wdi, nan=-np.inf), out=max_wdi)

max_wdi[nodata_mask | np.isinf(max_wdi)] = np.nan

print("\n[5/6] 归一化WDI...")

all_valid_wdi = np.concatenate([v[~np.isnan(v)] for v in year_raw_wdi.values()])
all_valid_wdi = all_valid_wdi[all_valid_wdi < 1e6]
wdi_lo    = float(np.percentile(all_valid_wdi, 0.5))
wdi_hi    = float(np.percentile(all_valid_wdi, 99.5))
wdi_scale = wdi_hi - wdi_lo + 1e-8

for yr, rw in year_raw_wdi.items():
    valid_yr = ~np.isnan(rw)
    yr_norm  = np.full_like(rw, np.nan, dtype=np.float32)
    yr_norm[valid_yr] = (rw[valid_yr] - wdi_lo) / wdi_scale
    yr_norm  = np.clip(yr_norm, 0.0, 1.0)
    np.save(os.path.join(OUTPUT_DIR, f'WDI_{yr}.npy'), yr_norm)

valid_max    = ~np.isnan(max_wdi)
max_wdi_norm = np.full_like(max_wdi, np.nan, dtype=np.float32)
max_wdi_norm[valid_max] = (max_wdi[valid_max] - wdi_lo) / wdi_scale
max_wdi_norm = np.clip(max_wdi_norm, 0.0, 1.0)

print("\n[6/6] 保存WDI输出...")

with rasterio.open(DEM_PATH) as ref:
    out_profile = ref.profile.copy()
out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='deflate')

data_out = np.where(np.isnan(max_wdi_norm), -9999.0, max_wdi_norm).astype(np.float32)
try:
    with rasterio.open(os.path.join(OUTPUT_DIR,'WDI_MultiYear_Max.tif'),
                       'w', **out_profile) as dst:
        dst.write(data_out, 1)
    print("    ✅ WDI_MultiYear_Max.tif")
except Exception as e:
    print(f"    ⚠️  TIF写入失败: {e}")

np.save(os.path.join(OUTPUT_DIR, 'WDI_MultiYear_Max.npy'), max_wdi_norm)
print("    ✅ WDI_MultiYear_Max.npy")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("Step 2 (v7.0) 完成！")
print("=" * 70)
print(f"  多年均值文件: Precipitation_Mean.npy, CR_MultiYear_Mean.npy")
print(f"\n  ★ v7.0新增逐年文件（{saved_rain}个年份）:")
for yr in sorted(year_rain_grids.keys()):
    r_ok = os.path.exists(os.path.join(OUTPUT_DIR, f'Precipitation_{yr}.npy'))
    c_ok = os.path.exists(os.path.join(OUTPUT_DIR, f'CR_{yr}.npy'))
    print(f"    {'✅' if r_ok else '❌'} Precipitation_{yr}.npy  "
          f"{'✅' if c_ok else '❌'} CR_{yr}.npy")