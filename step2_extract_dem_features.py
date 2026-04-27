"""
Step 2: DEM水文特征提取（最终合并版）
======================================
整合了 step2_extract_dem_features.py 和 fix_slope_twi_hand.py
路径已更新为本地实际路径

输入:
  - F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif
  - F:\Data\src\fac\Aligned_Soil_Params\K_SCH_Aligned_30m.tif
  - F:\Data\src\fac\Aligned_Soil_Params\PSI_Aligned_30m.tif
  - F:\Data\src\fac\Aligned_Soil_Params\THSCH_Aligned_30m.tif
  - F:\Data\src\files\flood_labels_clean.csv

输出: Step2/dem_soil_features.csv
      列: latitude, longitude, dem_m, slope_deg, TWI, HAND_m,
          ks_mmh, psi_cm, theta_s, IC_max_mmh
"""

import numpy as np
import pandas as pd
import rasterio
from pysheds.grid import Grid
from scipy.ndimage import sobel as scipy_sobel, minimum_filter
import os, warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
DEM_PATH   = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
KS_PATH    = r'F:\Data\src\fac\Aligned_Soil_Params\K_SCH_Aligned_30m.tif'
PSI_PATH   = r'F:\Data\src\fac\Aligned_Soil_Params\PSI_Aligned_30m.tif'
THS_PATH   = r'F:\Data\src\fac\Aligned_Soil_Params\THSCH_Aligned_30m.tif'
LABEL_CSV  = r'F:\Data\src\files\flood_labels_clean.csv'
OUTPUT_DIR = r'.\Step2'
TMP_DIR    = r'.\Step2\tmp'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ==================== 读取点位 ====================
labels = pd.read_csv(LABEL_CSV)
sites  = labels[['latitude','longitude']].drop_duplicates().reset_index(drop=True)
lats, lons = sites['latitude'].values, sites['longitude'].values
print(f"提取特征的点位数: {len(sites)}")

# ==================== 工具函数 ====================
def sample_raster(tif_path, lats, lons):
    """rasterio最近邻采样（正确的坐标→像素转换）"""
    with rasterio.open(tif_path) as src:
        nodata, data = src.nodata, src.read(1)
        h, w = src.height, src.width
        vals = []
        for lat, lon in zip(lats, lons):
            r, c = src.index(lon, lat)
            r, c = max(0, min(int(r), h-1)), max(0, min(int(c), w-1))
            v = float(data[r, c])
            if nodata is not None and abs(v - nodata) < 1e-3:
                v = np.nan
            vals.append(v)
    return np.array(vals)

def save_arr_as_tif(arr, ref_tif, out_path):
    """numpy数组→GeoTIF，继承ref_tif的地理参考"""
    with rasterio.open(ref_tif) as ref:
        profile = ref.profile.copy()
    profile.update(dtype='float32', count=1, nodata=-9999.0, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        out = arr.astype(np.float32)
        out = np.where(np.isnan(out), -9999.0, out)
        dst.write(out, 1)

def save_and_sample(arr, ref_tif, lats, lons, tmp_path):
    """保存为TIF再用rasterio采样——彻底解决pysheds坐标转换错位问题"""
    save_arr_as_tif(arr, ref_tif, tmp_path)
    vals = sample_raster(tmp_path, lats, lons)
    return np.where(vals == -9999.0, np.nan, vals)

# ==================== [1] 静态土壤参数 ====================
print("\n[1/4] 提取土壤水文参数...")
ks_vals  = sample_raster(KS_PATH,  lats, lons)
psi_vals = sample_raster(PSI_PATH, lats, lons)
ths_vals = sample_raster(THS_PATH, lats, lons)

nan_ks  = np.isnan(ks_vals).sum()
nan_psi = np.isnan(psi_vals).sum()
nan_ths = np.isnan(ths_vals).sum()

# 城市不透水层NaN补偿（模拟硬化路面）
ks_vals  = np.where(np.isnan(ks_vals),  0.01,  ks_vals)
psi_vals = np.where(np.isnan(psi_vals), -10.0, psi_vals)
ths_vals = np.where(np.isnan(ths_vals), 0.10,  ths_vals)

print(f"   Ks  : {ks_vals.min():.2f}~{ks_vals.max():.2f} mm/h  (NaN补偿{nan_ks}个)")
print(f"   Psi : {psi_vals.min():.2f}~{psi_vals.max():.2f} cm  (NaN补偿{nan_psi}个)")
print(f"   θs  : {ths_vals.min():.3f}~{ths_vals.max():.3f} m³/m³  (NaN补偿{nan_ths}个)")

# ==================== [2] DEM高程 ====================
print("\n[2/4] 提取DEM高程...")
dem_vals = sample_raster(DEM_PATH, lats, lons)
print(f"   DEM : {dem_vals.min():.1f}~{dem_vals.max():.1f} m")
print(f"   高程<50m的低洼点: {(dem_vals<50).sum()}个")

# ==================== [3] pysheds 水文特征 ====================
print("\n[3/4] pysheds水文特征（约15~25分钟）...")

# --- 读取原始DEM，Sobel坡度用原始DEM（保留真实地形起伏）---
print("   读取DEM并计算Sobel坡度...")
grid = Grid.from_raster(DEM_PATH)
dem  = grid.read_raster(DEM_PATH)
dem_raw = np.array(dem, dtype=np.float32)

res_deg = 0.00027778
res_m_x = res_deg * 111320 * np.cos(np.deg2rad(40.0))  # ≈23.6m（经向）
res_m_y = res_deg * 111320                               # ≈30.9m（纬向）
dem_clean = np.where(dem_raw < -100, np.nan, dem_raw)
dz_dy = scipy_sobel(dem_clean, axis=0) / (8 * res_m_y)
dz_dx = scipy_sobel(dem_clean, axis=1) / (8 * res_m_x)
slope_arr = np.clip(np.rad2deg(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))), 0, 60)
print("   Sobel坡度完成")

# --- fill洼地 → D8流向 → 汇流累积 ---
print("   DEM fill预处理（约3~5分钟）...")
pit_filled = grid.fill_pits(dem)
flooded    = grid.fill_depressions(pit_filled)
inflated   = grid.resolve_flats(flooded)
print("   DEM预处理完成")

dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
fdir   = grid.flowdir(inflated, dirmap=dirmap)
print("   D8流向完成")

acc     = grid.accumulation(fdir, dirmap=dirmap)
acc_arr = np.array(acc, dtype=np.float32)
print("   汇流累积完成")

# --- TWI = ln(汇流面积 / tan(坡度)) ---
cell_area  = res_m_x * res_m_y               # 实际像元面积（约730m²）
flow_area  = (acc_arr + 1) * cell_area
slope_prot = np.where(slope_arr < 0.1, 0.1, slope_arr)  # 最小坡度兜底
twi_arr    = np.clip(np.log(flow_area / np.tan(np.deg2rad(slope_prot))), 0, 25)
print("   TWI完成")

# --- HAND（传Raster对象，修复版）---
print("   计算HAND（约5~10分钟）...")
HAND_METHOD = None

try:
    channel_mask = acc > 500          # 汇流>500像元定义为河道（Raster布尔运算）
    hand_raster  = grid.compute_hand(fdir, inflated, channel_mask, dirmap=dirmap)
    hand_np      = np.array(hand_raster, dtype=np.float32)
    nan_ratio    = np.isnan(hand_np).mean()
    print(f"   compute_hand完成，NaN比例: {nan_ratio*100:.1f}%")

    if nan_ratio > 0.5:
        raise ValueError(f"NaN过多({nan_ratio*100:.0f}%)")

    hand_arr    = np.clip(np.where(np.isnan(hand_np), 0, hand_np), 0, 500)
    HAND_METHOD = 'compute_hand'

except Exception as e:
    print(f"   compute_hand失败({e})，使用备用方案...")
    dem_f    = np.where(np.isnan(dem_clean), np.nanmean(dem_clean), dem_clean)
    hand_sm  = np.clip(dem_f - minimum_filter(dem_f, size=17), 0, 500)
    hand_lg  = np.clip(dem_f - minimum_filter(dem_f, size=51), 0, 500)
    hand_arr = (hand_sm + hand_lg) / 2
    HAND_METHOD = 'local_min_dual'
    print(f"   备用HAND完成（{HAND_METHOD}）")

print(f"   HAND完成（方法: {HAND_METHOD}）")

# ==================== [4] 保存TIF → rasterio采样 ====================
# 核心修复：pysheds的grid.extent坐标系与经纬度不匹配
# 解决方案：先将numpy数组保存为继承DEM地理参考的TIF，再用rasterio采样
print("\n[4/4] 保存临时TIF并采样（v4坐标修复）...")

slope_vals = save_and_sample(slope_arr, DEM_PATH, lats, lons,
                             os.path.join(TMP_DIR, 'slope_tmp.tif'))
acc_vals   = save_and_sample(acc_arr,   DEM_PATH, lats, lons,
                             os.path.join(TMP_DIR, 'acc_tmp.tif'))
twi_vals   = save_and_sample(twi_arr,   DEM_PATH, lats, lons,
                             os.path.join(TMP_DIR, 'twi_tmp.tif'))
hand_vals  = save_and_sample(hand_arr,  DEM_PATH, lats, lons,
                             os.path.join(TMP_DIR, 'hand_tmp.tif'))

# NaN填充（中位数兜底）
for name, arr in [('slope',slope_vals),('TWI',twi_vals),
                  ('HAND',hand_vals),  ('acc',acc_vals)]:
    n = np.isnan(arr).sum()
    if n > 0:
        print(f"   ⚠️ {name}含{n}个NaN，用中位数填充")
        arr[:] = np.where(np.isnan(arr), np.nanmedian(arr), arr)

slope_vals = np.clip(slope_vals, 0, 60)
hand_vals  = np.clip(hand_vals,  0, 500)

print(f"   坡度(°): {slope_vals.min():.4f}~{slope_vals.max():.4f}")
print(f"   汇流   : {acc_vals.min():.0f}~{acc_vals.max():.0f}")
print(f"   TWI    : {twi_vals.min():.3f}~{twi_vals.max():.3f}")
print(f"   HAND(m): {hand_vals.min():.2f}~{hand_vals.max():.2f}")

# ==================== [5] 合并保存 ====================
df_out = pd.DataFrame({
    'latitude'   : lats,
    'longitude'  : lons,
    'dem_m'      : dem_vals.round(2),
    'slope_deg'  : slope_vals.round(4),
    'TWI'        : twi_vals.round(4),
    'HAND_m'     : hand_vals.round(2),
    'ks_mmh'     : ks_vals.round(4),
    'psi_cm'     : psi_vals.round(4),
    'theta_s'    : ths_vals.round(4),
    'IC_max_mmh' : ks_vals.round(4),   # Green-Ampt入渗能力上限
})

out_path = os.path.join(OUTPUT_DIR, 'dem_soil_features.csv')
df_out.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"\n{'='*55}")
print(f"✅ Step2完成: {out_path}")
print(f"   点位数  : {len(df_out)}")
print(f"   HAND方法: {HAND_METHOD}")
print(f"   临时文件: {TMP_DIR}（可手动删除）")
print(f"\n描述统计（关键列）:")
print(df_out[['dem_m','slope_deg','TWI','HAND_m',
              'ks_mmh','psi_cm','theta_s']].describe().round(3).to_string())