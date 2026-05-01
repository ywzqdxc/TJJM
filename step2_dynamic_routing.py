"""
Step 2 (v5.0): 动态水文物理汇流模型 - WDI计算 + 降雨统计
============================================================
v5.0 相对 v3.1 的改动（仅新增，原逻辑完全保留）：
  ★ 新增 year_rain_grids 字典，收集逐年汛期累积降雨栅格
  ★ 循环结束后保存 Precipitation_Mean.npy（多年汛期平均降雨，mm）
    → 作为 Step3 VSC框架的"降雨量"指标原始输入

原有输出完全保留：
  WDI_20XX.npy（逐年归一化WDI）
  WDI_MultiYear_Max.npy（多年极大值WDI）
  WDI_MultiYear_Max.tif
  CR_MultiYear_Mean.npy（由 step2_patch_cr_output.py 或本文件生成）
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
# 路径配置（与 v3.1 完全一致）
# ============================================================
NC_DIR     = r'E:\Data\China_Pre_1901_2024_from_CRUv4.09'
SM_DIR     = r'E:\Data\src\Processed_Decadal_SM_UTM50N'
DEM_PATH   = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
STATIC_DIR = r'./Step_New/Static'
OUTPUT_DIR = r'./Step_New/Dynamic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 产流系数配置（与 v3.1 完全一致）
# ============================================================
CR_URBAN   = 0.90
CR_ROCK    = 0.70

YEARS  = list(range(2012, 2025))
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

def get_decade(day):
    if day <= 10: return 1
    elif day <= 20: return 2
    else: return 3

print("=" * 70)
print("Step 2 (v5.0)：动态水文物理汇流模型")
print("新增输出: Precipitation_Mean.npy + CR_MultiYear_Mean.npy")
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

urban_count   = int(urban_mask.sum())
rock_count    = int(rock_mask.sum())
total_valid   = int(valid_mask.sum())
natural_count = total_valid - urban_count - rock_count

print(f"    栅格尺寸: {h} × {w}")
print(f"    全域有效像元: {total_valid:,}")
print(f"    ├─ 自然土壤区: {natural_count:,} ({natural_count/total_valid*100:.1f}%)")
print(f"    ├─ 城市建成区: {urban_count:,} ({urban_count/total_valid*100:.1f}%) → CR={CR_URBAN}")
print(f"    └─ 山区裸岩区: {rock_count:,} ({rock_count/total_valid*100:.1f}%) → CR={CR_ROCK}")

# 初始化 pysheds
grid         = Grid.from_raster(DEM_PATH)
base_raster  = grid.read_raster(DEM_PATH)
fdir_obj     = base_raster.copy()
fdir_obj[:]  = fdir_raw
fdir_obj.nodata = 0

# Ks归一化（与 v3.1 完全一致）
ks_nat  = ks[~np.isnan(ks) & ~urban_mask & ~rock_mask]
ks_norm = np.zeros_like(ks)
if ks_nat.size > 0:
    ks_norm = (ks - ks_nat.min()) / (ks_nat.max() - ks_nat.min() + 1e-6)
ks_norm = np.where(np.isnan(ks), 0.0, np.clip(ks_norm, 0, 1))

print(f"\n    Ks归一化: 有效={ks_nat.size:,}  "
      f"范围=[{ks_nat.min():.4f},{ks_nat.max():.4f}] mm/h")


# ============================================================
# 2. 逐年计算（★ 新增 year_rain_grids + year_cr_arrays）
# ============================================================
print("\n[2/6] 逐年计算汇流累积量（CHM_PRE V2，汛期6/15-9/15）...")

year_acc_raw    = {}
year_rain_grids = {}   # ★ v5.0新增：逐年汛期累积降雨 (mm)
year_cr_arrays  = {}   # ★ v5.0新增：逐年径流系数

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

    # 汛期过滤
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

    # 峰值旬土壤水分
    peak_date  = flood_dates[np.argmax(daily_mean)]
    decade_idx = get_decade(peak_date.day)
    sm_file    = os.path.join(SM_DIR, str(year),
                              f'SM_{year}_{peak_date.month:02d}_{decade_idx}_Mean_30m.tif')
    if os.path.exists(sm_file):
        with rasterio.open(sm_file) as src:
            initial_sm = src.read(1, out_shape=(h, w)).astype(np.float32)
        initial_sm = np.where(initial_sm < 0, 0.14, initial_sm)
        initial_sm = np.where(np.isnan(initial_sm), 0.25, initial_sm)  # NaN 用默认值填充
    else:
        initial_sm = np.full((h, w), 0.25, dtype=np.float32)

    # 全汛期累积降雨重采样
    season_rain_nc = flood_rain.sum(dim='time').values
    h_r, w_r = season_rain_nc.shape
    scale_y, scale_x = h / h_r, w / w_r
    if abs(scale_y - 1.0) < 1e-6 and abs(scale_x - 1.0) < 1e-6:
        rain_grid = season_rain_nc
    else:
        rain_grid = zoom(season_rain_nc, (scale_y, scale_x), order=1)
    season_rain_m = np.clip(rain_grid, 0, 3000)

    # ── 产流系数计算（与 v3.1 完全一致）────────────────
    soil_deficit = np.clip(ths - initial_sm, 0.0, None)
    soil_deficit[urban_mask] = 0.0
    soil_deficit[rock_mask]  = 0.0
    infil_factor = np.clip(soil_deficit * ks_norm * 2.0, 0.0, 0.70)
    runoff_coeff = np.clip(0.9 - infil_factor, 0.05, 0.95)
    runoff_coeff[urban_mask] = CR_URBAN
    runoff_coeff[rock_mask]  = CR_ROCK
    # ─────────────────────────────────────────────────────

    effective_runoff = season_rain_m * runoff_coeff

    # pysheds 加权汇流
    weights_raster    = base_raster.copy()
    weights_raster[:] = effective_runoff
    weights_raster.nodata = 0

    try:
        acc_water = grid.accumulation(fdir_obj, dirmap=DIRMAP, weights=weights_raster)
        acc_arr   = np.array(acc_water, dtype=np.float32)
        year_acc_raw[year] = acc_arr

        # ★ v5.0新增：保存降雨和CR
        year_rain_grids[year] = season_rain_m.astype(np.float32)
        rc_save = runoff_coeff.copy().astype(np.float32)
        rc_save[nodata_mask] = np.nan
        year_cr_arrays[year] = rc_save

        rain_valid = season_rain_m[valid_mask]
        print(f"✓ ({n_days}天)  雨量均值={rain_valid.mean():.1f}mm  "
              f"CR均值={runoff_coeff[valid_mask].mean():.3f}")
    except Exception as e:
        print(f"✗ 汇流失败: {e}")

    ds.close()

print(f"\n  成功计算: {len(year_acc_raw)}/{len(YEARS)} 年")


# ============================================================
# ★ 2.5  保存降雨量和CR多年均值（v5.0新增）
# ============================================================
print("\n[★ 2.5] 保存 Precipitation_Mean.npy & CR_MultiYear_Mean.npy...")

if year_rain_grids:
    rain_stack  = np.stack(list(year_rain_grids.values()), axis=0)
    rain_mean   = np.nanmean(rain_stack, axis=0).astype(np.float32)
    rain_mean   = np.where(nodata_mask, np.nan, rain_mean)
    rain_path   = os.path.join(OUTPUT_DIR, 'Precipitation_Mean.npy')
    np.save(rain_path, rain_mean)
    v_r = rain_mean[valid_mask & ~np.isnan(rain_mean)]
    print(f"  ✅ Precipitation_Mean.npy")
    print(f"     有效={v_r.size:,}  均值={v_r.mean():.1f}mm  "
          f"范围=[{v_r.min():.1f},{v_r.max():.1f}]mm  "
          f"可用年份={sorted(year_rain_grids.keys())}")
else:
    print("  ⚠️  无降雨数据，Precipitation_Mean.npy 未生成")

if year_cr_arrays:
    cr_stack = np.stack(list(year_cr_arrays.values()), axis=0)
    cr_mean  = np.nanmean(cr_stack, axis=0).astype(np.float32)
    cr_mean  = np.where(nodata_mask, np.nan, cr_mean)
    cr_path  = os.path.join(OUTPUT_DIR, 'CR_MultiYear_Mean.npy')
    np.save(cr_path, cr_mean)
    v_c = cr_mean[valid_mask & ~np.isnan(cr_mean)]
    print(f"  ✅ CR_MultiYear_Mean.npy")
    print(f"     有效={v_c.size:,}  均值={v_c.mean():.4f}  "
          f"范围=[{v_c.min():.4f},{v_c.max():.4f}]")
else:
    print("  ⚠️  无CR数据，CR_MultiYear_Mean.npy 未生成")


# ============================================================
# 3. 全局缩放因子计算（与 v3.1 完全一致）
# ============================================================
print("\n[3/6] 计算全局WDI缩放因子...")

_chunks     = [arr[~nodata_mask & (arr > 1e-12)] for arr in year_acc_raw.values()]
_chunks     = [c for c in _chunks if c.size > 0]
all_acc_pos = np.concatenate(_chunks) if _chunks else np.array([])

if all_acc_pos.size > 100:
    global_acc_p75 = float(np.percentile(all_acc_pos, 75))
    global_scale   = (np.exp(6.0) - 1.0) / max(global_acc_p75, 1e-12)
    print(f"    全局acc P75 = {global_acc_p75:.6e}")
    print(f"    全局缩放因子 Sg = {global_scale:.4e}")
else:
    global_scale = 1e8
    print("    ⚠️  有效acc数据不足，使用默认缩放因子")


# ============================================================
# 4. 计算各年WDI（与 v3.1 完全一致）
# ============================================================
print("\n[4/6] 计算各年WDI并求多年最大值...")

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


# ============================================================
# 5. 归一化至[0,1]（与 v3.1 完全一致）
# ============================================================
print("\n[5/6] 归一化WDI至[0,1]...")

all_valid = np.concatenate([v[~np.isnan(v)] for v in year_raw_wdi.values()])
all_valid = all_valid[all_valid < 1e6]
wdi_lo    = float(np.percentile(all_valid, 0.5))
wdi_hi    = float(np.percentile(all_valid, 99.5))
wdi_scale = wdi_hi - wdi_lo + 1e-8
print(f"    WDI原始范围: [{wdi_lo:.4f}, {wdi_hi:.4f}]")

# 逐年保存
for yr, rw in year_raw_wdi.items():
    valid_yr = ~np.isnan(rw)
    yr_norm  = np.full_like(rw, np.nan, dtype=np.float32)
    yr_norm[valid_yr] = (rw[valid_yr] - wdi_lo) / wdi_scale
    yr_norm  = np.clip(yr_norm, 0.0, 1.0)
    np.save(os.path.join(OUTPUT_DIR, f'WDI_{yr}.npy'), yr_norm)

# 多年极大值
valid_max    = ~np.isnan(max_wdi)
max_wdi_norm = np.full_like(max_wdi, np.nan, dtype=np.float32)
max_wdi_norm[valid_max] = (max_wdi[valid_max] - wdi_lo) / wdi_scale
max_wdi_norm = np.clip(max_wdi_norm, 0.0, 1.0)


# ============================================================
# 6. 保存WDI输出（与 v3.1 完全一致）
# ============================================================
print("\n[6/6] 保存WDI输出文件...")

with rasterio.open(DEM_PATH) as ref:
    out_profile = ref.profile.copy()
out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='deflate')

out_tif  = os.path.join(OUTPUT_DIR, 'WDI_MultiYear_Max.tif')
data_out = np.where(np.isnan(max_wdi_norm), -9999.0, max_wdi_norm).astype(np.float32)
try:
    with rasterio.open(out_tif, 'w', **out_profile) as dst:
        dst.write(data_out, 1)
    print(f"    ✅ WDI_MultiYear_Max.tif")
except Exception as e:
    print(f"    ⚠️  TIF写入失败: {e}")

np.save(os.path.join(OUTPUT_DIR, 'WDI_MultiYear_Max.npy'), max_wdi_norm)
print(f"    ✅ WDI_MultiYear_Max.npy")


# ============================================================
# 验证与统计
# ============================================================
print("\n" + "=" * 70)
print("Step 2 (v5.0) 完成！验证输出：")
print("=" * 70)

wdi_valid = max_wdi_norm[valid_mask & ~np.isnan(max_wdi_norm)]
print(f"  WDI 有效像元: {len(wdi_valid):,}")
print(f"  WDI 均值={np.nanmean(wdi_valid):.4f}  "
      f"P50={np.nanpercentile(wdi_valid,50):.4f}  "
      f"P95={np.nanpercentile(wdi_valid,95):.4f}")

print(f"\n  ★ v5.0新增输出:")
for fname in ['Precipitation_Mean.npy', 'CR_MultiYear_Mean.npy']:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        arr = np.load(fpath)
        v   = arr[valid_mask & ~np.isnan(arr)]
        print(f"    ✅ {fname}  均值={v.mean():.4f}  "
              f"范围=[{v.min():.4f},{v.max():.4f}]")
    else:
        print(f"    ❌ {fname} 未生成！")

print(f"\n  WDI逐年文件:")
for yr in YEARS:
    fp = os.path.join(OUTPUT_DIR, f'WDI_{yr}.npy')
    st = "✅" if os.path.exists(fp) else "❌"
    print(f"    {st} WDI_{yr}.npy")