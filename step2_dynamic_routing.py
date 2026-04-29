"""
紧急修复：从Step2内存数据重建WDI_MultiYear_Max.tif
==================================================
直接运行step2_dynamic_routing.py的核心计算部分，生成有效的WDI TIFF
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
# 路径配置（与你step2完全相同）
# ============================================================
NC_DIR     = r'F:\Data\China_Pre_1901_2024_from_CRUv4.09'
SM_DIR     = r'F:\Data\src\Processed_Decadal_SM_UTM50N'
DEM_PATH   = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
STATIC_DIR = r'./Step_New/Static'
OUTPUT_DIR = r'./Step_New/Dynamic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS  = list(range(2012, 2025))
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

def get_decade(day):
    if day <= 10: return 1
    elif day <= 20: return 2
    else: return 3

print("=" * 70)
print("WDI 紧急重建脚本")
print("=" * 70)

# 1. 加载静态数据
print("\n[1/6] 加载静态数据...")
fdir_raw    = np.load(os.path.join(STATIC_DIR, 'fdir.npy'))
slope       = np.load(os.path.join(STATIC_DIR, 'slope.npy'))
hand        = np.load(os.path.join(STATIC_DIR, 'hand.npy'))
ks          = np.load(os.path.join(STATIC_DIR, 'ks.npy'))
ths         = np.load(os.path.join(STATIC_DIR, 'ths.npy'))
urban_mask  = np.load(os.path.join(STATIC_DIR, 'urban_mask.npy'))
nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy'))

h, w = fdir_raw.shape
valid_mask = ~nodata_mask

# 初始化pysheds
grid = Grid.from_raster(DEM_PATH)
base_raster = grid.read_raster(DEM_PATH)
fdir_obj = base_raster.copy()
fdir_obj[:] = fdir_raw
fdir_obj.nodata = 0

# 归一化Ks
ks_nat = ks[~np.isnan(ks) & ~urban_mask]
ks_norm = np.zeros_like(ks)
if ks_nat.size > 0:
    ks_norm = (ks - ks_nat.min()) / (ks_nat.max() - ks_nat.min() + 1e-6)
ks_norm = np.where(np.isnan(ks), 0.0, np.clip(ks_norm, 0, 1))

print(f"    栅格尺寸: {h} × {w}")

# 2. 逐年计算acc
print("\n[2/6] 逐年计算累积汇流量...")
year_acc_raw = {}

for year in YEARS:
    nc_path = os.path.join(NC_DIR, f'CHM_PRE_V2_daily_{year}.nc')
    if not os.path.exists(nc_path):
        nc_path = os.path.join(NC_DIR, f'{year}.nc')
    if not os.path.exists(nc_path):
        print(f"  {year}: 跳过（无降雨文件）")
        continue

    print(f"  {year}...", end=' ', flush=True)

    ds = xr.open_dataset(nc_path, engine='netcdf4')
    var_name = 'prec' if 'prec' in ds.data_vars else list(ds.data_vars)[0]
    rain_data = ds[var_name].sel(lat=slice(39.4, 41.1), lon=slice(115.4, 117.5))
    all_dates = pd.to_datetime(ds['time'].values)

    # 汛期过滤
    flood_mask = (
        ((all_dates.month > 6) | ((all_dates.month == 6) & (all_dates.day >= 15))) &
        ((all_dates.month < 9) | ((all_dates.month == 9) & (all_dates.day <= 15)))
    )

    if not flood_mask.any():
        print("无汛期数据")
        continue

    flood_rain = rain_data[flood_mask]
    flood_dates = all_dates[flood_mask]
    daily_mean = flood_rain.mean(dim=['lat', 'lon']).values
    n_days = int(flood_mask.sum())

    # 峰值旬土壤水分
    peak_date = flood_dates[np.argmax(daily_mean)]
    decade_idx = get_decade(peak_date.day)
    sm_file = os.path.join(SM_DIR, str(year),
                           f'SM_{year}_{peak_date.month:02d}_{decade_idx}_Mean_30m.tif')

    if os.path.exists(sm_file):
        with rasterio.open(sm_file) as src:
            initial_sm = src.read(1, out_shape=(h, w)).astype(np.float32)
        initial_sm = np.where(initial_sm < 0, 0.14, initial_sm)
    else:
        initial_sm = np.full((h, w), 0.25, dtype=np.float32)

    # 累积降雨重采样
    season_rain_nc = flood_rain.sum(dim='time').values
    h_rain, w_rain = season_rain_nc.shape
    scale_y, scale_x = h / h_rain, w / w_rain

    if abs(scale_y - 1.0) < 1e-6 and abs(scale_x - 1.0) < 1e-6:
        rain_grid = season_rain_nc
    else:
        rain_grid = zoom(season_rain_nc, (scale_y, scale_x), order=1)

    season_rain_m = np.clip(rain_grid, 0, 3000)

    # 产流计算
    soil_deficit = np.clip(ths - initial_sm, 0.0, None)
    soil_deficit[urban_mask] = 0.0
    infil_factor = np.clip(soil_deficit * ks_norm * 2.0, 0.0, 0.70)
    runoff_coeff = np.clip(0.9 - infil_factor, 0.05, 0.95)
    runoff_coeff[urban_mask] = 0.90
    effective_runoff = season_rain_m * runoff_coeff

    # pysheds汇流
    weights_raster = base_raster.copy()
    weights_raster[:] = effective_runoff
    weights_raster.nodata = 0

    try:
        acc_water = grid.accumulation(fdir_obj, dirmap=DIRMAP, weights=weights_raster)
        acc_arr = np.array(acc_water, dtype=np.float64)
        year_acc_raw[year] = acc_arr.astype(np.float32)
        print(f"✓ ({n_days}天, 累计{n_days}天)")
    except Exception as e:
        print(f"✗ 汇流失败: {e}")

print(f"\n  成功计算: {len(year_acc_raw)}/{len(YEARS)} 年")

# 3. 全局scale计算
print("\n[3/6] 计算全局WDI...")
_chunks = [arr[~nodata_mask & (arr > 1e-12)] for arr in year_acc_raw.values()]
_chunks = [c for c in _chunks if c.size > 0]
all_acc_pos = np.concatenate(_chunks) if _chunks else np.array([])

if all_acc_pos.size > 100:
    global_acc_p75 = float(np.percentile(all_acc_pos, 75))
    global_scale = (np.exp(6.0) - 1.0) / max(global_acc_p75, 1e-12)
    print(f"    全局acc P75={global_acc_p75:.6e}  全局scale={global_scale:.3e}")
else:
    global_scale = 1e8

# 预计算分母
slope_s = np.clip(slope, 0.5, 60.0).astype(np.float64)
hand_s = np.clip(hand, 0.0, 500.0).astype(np.float64)
log_denom = np.log1p(slope_s) * np.log1p(hand_s)

# 4. 计算各年WDI并累积最大值
print("\n[4/6] 计算各年WDI并求多年最大值...")
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
    # 更新最大值
    np.maximum(max_wdi, np.nan_to_num(raw_wdi, nan=-np.inf), out=max_wdi)

max_wdi[nodata_mask | np.isinf(max_wdi)] = np.nan

# 5. 归一化到[0,1]
print("\n[5/6] 归一化WDI...")
all_valid = np.concatenate([v[~np.isnan(v)] for v in year_raw_wdi.values()])
all_valid = all_valid[all_valid < 1e6]  # 过滤异常值

wdi_lo = float(np.percentile(all_valid, 0.5))
wdi_hi = float(np.percentile(all_valid, 99.5))
wdi_scale = wdi_hi - wdi_lo + 1e-8

print(f"    WDI范围: [{wdi_lo:.4f}, {wdi_hi:.4f}]")

# 归一化多年最大值
valid_max = ~np.isnan(max_wdi)
max_wdi_norm = np.full_like(max_wdi, np.nan, dtype=np.float32)
max_wdi_norm[valid_max] = (max_wdi[valid_max] - wdi_lo) / wdi_scale
max_wdi_norm = np.clip(max_wdi_norm, 0.0, 1.0)

# 6. 保存（确保成功）
print("\n[6/6] 保存WDI TIFF...")

with rasterio.open(DEM_PATH) as ref:
    out_profile = ref.profile.copy()
out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='deflate')

out_tif = os.path.join(OUTPUT_DIR, 'WDI_MultiYear_Max.tif')
data_out = np.where(np.isnan(max_wdi_norm), -9999.0, max_wdi_norm).astype(np.float32)

# 方法1：正常写入
try:
    with rasterio.open(out_tif, 'w', **out_profile) as dst:
        dst.write(data_out, 1)
    print(f"    ✅ 写入成功: {out_tif}")
except Exception as e:
    print(f"    ⚠️  标准写入失败: {e}")

    # 方法2：分块写入
    print("    尝试分块写入...")
    try:
        with rasterio.open(out_tif, 'w', **out_profile) as dst:
            chunk_size = 500
            for i in range(0, h, chunk_size):
                i_end = min(i + chunk_size, h)
                dst.write(data_out[i:i_end, :], 1, window=((i, i_end), (0, w)))
        print(f"    ✅ 分块写入成功: {out_tif}")
    except Exception as e2:
        print(f"    ❌ 分块写入也失败: {e2}")

        # 方法3：保存为NPY（最后手段）
        out_npy = os.path.join(OUTPUT_DIR, 'WDI_MultiYear_Max.npy')
        np.save(out_npy, max_wdi_norm)
        print(f"    ✅ 已保存NPY: {out_npy}")

# 同时保存NPY备份
npy_backup = os.path.join(OUTPUT_DIR, 'WDI_MultiYear_Max.npy')
np.save(npy_backup, max_wdi_norm)
print(f"    ✅ NPY备份: {npy_backup}")

# 验证
print("\n" + "=" * 70)
print("验证结果:")
print("=" * 70)
wdi_valid = max_wdi_norm[valid_mask & ~np.isnan(max_wdi_norm)]
print(f"  有效像元: {len(wdi_valid):,}")
print(f"  均值: {np.nanmean(wdi_valid):.4f}")
print(f"  P25:  {np.nanpercentile(wdi_valid, 25):.4f}")
print(f"  P50:  {np.nanpercentile(wdi_valid, 50):.4f}")
print(f"  P75:  {np.nanpercentile(wdi_valid, 75):.4f}")
print(f"  P95:  {np.nanpercentile(wdi_valid, 95):.4f}")
print(f"  P99:  {np.nanpercentile(wdi_valid, 99):.4f}")
print(f"  CV:   {np.nanstd(wdi_valid)/np.nanmean(wdi_valid):.2f}")

# 验证文件可读性
print(f"\n  验证TIFF可读性...")
try:
    with rasterio.open(out_tif) as test:
        test_data = test.read(1)
        print(f"  ✅ TIFF可读取，shape={test_data.shape}")
except:
    print(f"  ⚠️  TIFF仍有问题，但NPY可用")
    print(f"  → 在step2_5_poi_exposure.py中使用NPY文件")

print(f"\n✅ 重建完成！")
print(f"   输出文件: {OUTPUT_DIR}")