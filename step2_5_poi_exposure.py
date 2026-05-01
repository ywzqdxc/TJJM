"""
Step 2.5b (修订版 v2.0): 外部社会脆弱性指标预处理
=====================================================
基于数据探查结果的精确修订：

关键修订点：
  1. DEM坐标系为 EPSG:4326（地理坐标，度），非投影坐标系
     → POI核密度带宽改用度为单位；路网密度先投影再计算
  2. 积水点时间格式5种，逐一处理：
     Timestamp对象(2018) / "M月D日"字符串 / Excel序列号(2022) / "M月D日\nH点M分"(2024/25)
  3. OSM路径为圆括号 (13-23)，目录名两位数字
  4. ~$临时文件跳过
  5. 人口2024 nodata=-32768，Albers投影CRS字符串直接用
  6. 避难场所CSV编码为gbk

输出（全部投影至DEM基准网格，归一化[0,1]）：
  ./Step_New/External/
    hospital_density_30m.tif
    shelter_density_30m.tif
    firestation_density_30m.tif
    population_density_30m.tif
    road_density_30m.tif
    waterlogging_point_density_30m.tif
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
from scipy.ndimage import gaussian_filter
import os, re, warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置（已根据探查结果修正）
# ============================================================
DEM_PATH        = r'E:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'
OUTPUT_DIR      = r'./Step_New/External'
STATIC_DIR      = r'./Step_New/Static'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# POI（编码已确认）
POI_HOSPITAL    = r'E:\Data\应对能力\医院POI.csv'           # utf-8
POI_SHELTER     = r'E:\Data\应对能力\应急避难场所POI.csv'   # gbk  ← 已修正
POI_FIRESTATION = r'E:\Data\应对能力\消防部门POI.CSV'       # utf-8

# 人口
POP_DIR         = r'E:\Data\人口密度数据'                   # 北京市_2012.tif ~ 北京市_2023.tif
POP_2024        = r'E:\Data\人口密度数据\北京市(24)'        # 特殊Albers格式
POP_NODATA      = -2147483647                               # int32 nodata
POP_NODATA_2024 = -32768                                    # int16 nodata（已确认）

# OSM（圆括号，两位年份目录）
ROAD_BASE_DIR   = r'E:\Data\全国各区县路网密度和总长度数据(13-23)'  # ← 已修正为圆括号

# 积水点
WATERLOG_DIR    = r'E:\Data\src\积水点位\积水点位'

STUDY_YEARS     = list(range(2012, 2025))

# ============================================================
# 主要道路类型（排除人行道/自行车道）
# ============================================================
ROAD_TYPES_KEEP = {
    'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'tertiary_link', 'residential', 'living_street',
    'unclassified', 'road', 'service'
}

print("=" * 70)
print("Step 2.5b (v2.0): 外部社会指标预处理")
print("=" * 70)


# ============================================================
# 一、加载基准网格
# ============================================================
print("\n[基准网格] 加载DEM...")
with rasterio.open(DEM_PATH) as ref:
    h, w          = ref.height, ref.width
    dst_crs       = ref.crs          # EPSG:4326
    dst_transform = ref.transform
    dst_bounds    = ref.bounds
    out_profile   = ref.profile.copy()

out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='deflate')

# 像元尺寸（度）
cell_deg_x = abs(dst_transform[0])   # ≈ 0.000278°
cell_deg_y = abs(dst_transform[4])   # ≈ 0.000278°

# 研究区中心纬度，用于度→米换算
center_lat = (dst_bounds.top + dst_bounds.bottom) / 2.0   # ≈ 40.25°
meter_per_deg_lat = 111320.0                                # 纬度方向
meter_per_deg_lon = 111320.0 * np.cos(np.deg2rad(center_lat))  # 经度方向

# 像元面积（m²）
cell_area_m2 = (cell_deg_x * meter_per_deg_lon) * (cell_deg_y * meter_per_deg_lat)

print(f"  DEM: {h}×{w}  CRS={dst_crs}  "
      f"分辨率={cell_deg_x:.6f}°×{cell_deg_y:.6f}°")
print(f"  像元等效面积: {cell_area_m2:.1f} m²  "
      f"({cell_deg_x*meter_per_deg_lon:.1f}m × {cell_deg_y*meter_per_deg_lat:.1f}m)")

# 加载有效掩膜
nodata_path = os.path.join(STATIC_DIR, 'nodata_mask.npy')
if os.path.exists(nodata_path):
    valid_mask = ~np.load(nodata_path).astype(bool)
    print(f"  nodata_mask: 有效像元={valid_mask.sum():,}")
else:
    with rasterio.open(DEM_PATH) as src:
        dem_data = src.read(1)
        valid_mask = dem_data > -100
    print(f"  ⚠️  nodata_mask.npy未找到，从DEM生成: {valid_mask.sum():,}")


# ============================================================
# 工具函数
# ============================================================

def normalize_01(arr, valid_mask, plo=1, phi=99):
    """百分位截断归一化至[0,1]"""
    result = np.full_like(arr, np.nan, dtype=np.float32)
    vals   = arr[valid_mask & np.isfinite(arr)]
    if vals.size < 10:
        return result
    lo, hi = float(np.percentile(vals, plo)), float(np.percentile(vals, phi))
    if hi - lo < 1e-10:
        result[valid_mask] = 0.0
        return result
    normed = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    result[valid_mask & np.isfinite(arr)] = normed[valid_mask & np.isfinite(arr)]
    return result


def save_tif(arr_norm, path, profile, valid_mask):
    """保存归一化TIF（无效=-9999）"""
    data_out = np.where(~valid_mask | ~np.isfinite(arr_norm),
                        -9999.0, arr_norm).astype(np.float32)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data_out, 1)
    v = arr_norm[valid_mask & np.isfinite(arr_norm)]
    fname = os.path.basename(path)
    if v.size > 0:
        print(f"    ✅ {fname}  有效={v.size:,}  "
              f"均值={v.mean():.4f}  范围=[{v.min():.4f},{v.max():.4f}]")
    else:
        print(f"    ⚠️  {fname} 无有效数据！")


def read_poi_csv(path, encoding='utf-8'):
    """读取POI CSV，返回含lon/lat列的DataFrame（过滤北京范围）"""
    df = pd.read_csv(path, encoding=encoding)
    # 重命名经纬度列（探查已确认是'经度'/'纬度'）
    rename_map = {}
    for col in df.columns:
        cl = col.strip()
        if cl == '经度': rename_map[col] = 'lon'
        elif cl == '纬度': rename_map[col] = 'lat'
    df = df.rename(columns=rename_map)
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df = df.dropna(subset=['lon', 'lat'])
    # 北京范围过滤
    df = df[(df['lon'] >= 115.4) & (df['lon'] <= 117.6) &
            (df['lat'] >= 39.4)  & (df['lat'] <= 41.1)]
    return df


def poi_to_density(df_poi, h, w, dst_transform,
                   bandwidth_deg=0.009, weight_col=None):
    """
    POI点 → 高斯核密度栅格
    DEM为EPSG:4326，直接用度坐标计算像素位置
    bandwidth_deg: 带宽（度），0.009°≈1km
    """
    lons = df_poi['lon'].values
    lats = df_poi['lat'].values

    # 经纬度 → 像素行列（直接用仿射变换逆）
    inv = ~dst_transform
    cols_f, rows_f = inv * (lons, lats)
    rows_i = np.round(rows_f).astype(int)
    cols_i = np.round(cols_f).astype(int)

    in_bounds = (rows_i >= 0) & (rows_i < h) & (cols_i >= 0) & (cols_i < w)
    rows_i = rows_i[in_bounds]
    cols_i = cols_i[in_bounds]

    if weight_col and weight_col in df_poi.columns:
        weights = df_poi.iloc[np.where(in_bounds)[0]][weight_col].values
        weights = np.where(np.isfinite(weights), weights, 1.0)
    else:
        weights = np.ones(in_bounds.sum())

    density = np.zeros((h, w), dtype=np.float32)
    np.add.at(density, (rows_i, cols_i), weights)

    # 像素sigma = 带宽度数 / 像元尺寸度数
    sigma = bandwidth_deg / cell_deg_x
    density = gaussian_filter(density, sigma=sigma)
    return density


# ============================================================
# 二、应对能力 POI → 核密度
# ============================================================
print("\n[1/4] 应对能力 POI 核密度...")

poi_configs = [
    # (名称, CSV路径, 编码, 输出文件名, 带宽度数)
    ('医院',     POI_HOSPITAL,    'utf-8', 'hospital_density_30m.tif',    0.013),  # ≈1.5km
    ('避难场所', POI_SHELTER,     'gbk',   'shelter_density_30m.tif',     0.018),  # ≈2km
    ('消防站',   POI_FIRESTATION, 'utf-8', 'firestation_density_30m.tif', 0.018),  # ≈2km
]

for name, csv_path, enc, out_name, bw in poi_configs:
    print(f"\n  [{name}] {os.path.basename(csv_path)}  编码={enc}")
    try:
        df_poi = read_poi_csv(csv_path, encoding=enc)
        print(f"    有效POI: {len(df_poi):,} 个")
        density = poi_to_density(df_poi, h, w, dst_transform, bandwidth_deg=bw)
        density_norm = normalize_01(density, valid_mask)
        save_tif(density_norm, os.path.join(OUTPUT_DIR, out_name), out_profile, valid_mask)
    except Exception as e:
        print(f"    ❌ 失败: {e}")


# ============================================================
# 三、人口密度 → 重投影 + 多年均值
# ============================================================
print("\n[2/4] 人口密度重投影...")

def reproject_pop(src_path, h, w, dst_crs, dst_transform,
                  nodata_val, additional_invalid_threshold=-1e6):
    """
    将人口TIF重投影至目标网格（EPSG:4326 30m）
    处理大负数无效值，返回float32（无效=NaN）
    """
    with rasterio.open(src_path) as src:
        src_crs       = src.crs
        src_transform = src.transform
        src_nodata    = src.nodata if src.nodata is not None else nodata_val
        data          = src.read(1).astype(np.float64)

    # 屏蔽无效值
    invalid = (data == src_nodata) | (data < additional_invalid_threshold)
    data[invalid] = np.nan

    # 构建内存栅格
    tmp_nd = -9999.0
    data_f32 = np.where(np.isnan(data), tmp_nd, data).astype(np.float32)

    with MemoryFile() as mf:
        with mf.open(driver='GTiff',
                      height=data_f32.shape[0], width=data_f32.shape[1],
                      count=1, dtype=np.float32,
                      crs=src_crs, transform=src_transform,
                      nodata=tmp_nd) as mem:
            mem.write(data_f32, 1)
            dst = np.full((h, w), tmp_nd, dtype=np.float32)
            reproject(
                source=rasterio.band(mem, 1),
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=tmp_nd
            )

    result = np.where(dst == tmp_nd, np.nan, dst).astype(np.float32)
    return result


pop_arrays = {}

# 2012-2023
for year in range(2012, 2024):
    path = os.path.join(POP_DIR, f'北京市_{year}.tif')
    print(f"  {year}...", end=' ', flush=True)
    try:
        arr = reproject_pop(path, h, w, dst_crs, dst_transform, POP_NODATA)
        pop_arrays[year] = arr
        v = arr[valid_mask & np.isfinite(arr)]
        print(f"✓  有效={v.size:,}  均值={v.mean():.0f}")
    except Exception as e:
        print(f"✗ {e}")

# 2024（Albers投影，int16，nodata=-32768）
print(f"  2024（Albers特殊格式）...", end=' ', flush=True)
if os.path.exists(POP_2024):
    try:
        arr24 = reproject_pop(
            POP_2024, h, w, dst_crs, dst_transform,
            nodata_val=POP_NODATA_2024,
            additional_invalid_threshold=-1000
        )
        pop_arrays[2024] = arr24
        v = arr24[valid_mask & np.isfinite(arr24)]
        print(f"✓  有效={v.size:,}  均值={v.mean():.0f}")
    except Exception as e:
        print(f"✗ {e}  → 用2023代替")
        if 2023 in pop_arrays:
            pop_arrays[2024] = pop_arrays[2023].copy()
else:
    print(f"✗ 文件不存在  → 用2023代替")
    if 2023 in pop_arrays:
        pop_arrays[2024] = pop_arrays[2023].copy()

print(f"\n  可用年份: {sorted(pop_arrays.keys())}")

if pop_arrays:
    stack = np.stack(list(pop_arrays.values()), axis=0)
    pop_mean = np.nanmean(stack, axis=0).astype(np.float32)
    pop_norm = normalize_01(np.log1p(np.where(pop_mean < 0, 0, pop_mean)), valid_mask)
    save_tif(pop_norm, os.path.join(OUTPUT_DIR, 'population_density_30m.tif'),
             out_profile, valid_mask)

    # 逐年保存（供Step4时序分析）
    print(f"  逐年文件...")
    for yr, arr in pop_arrays.items():
        yr_norm = normalize_01(np.log1p(np.where(arr < 0, 0, arr)), valid_mask)
        save_tif(yr_norm, os.path.join(OUTPUT_DIR, f'population_{yr}_30m.tif'),
                 out_profile, valid_mask)
else:
    print("  ❌ 无可用人口数据！")


# ============================================================
# 四、OSM 路网 → 像元级密度
# ============================================================
print("\n[3/4] OSM 路网密度...")

# 年份→目录（2012→13, 2024→23）
def year_to_folder(year):
    if year < 2013: return '13'
    if year > 2023: return '23'
    return str(year)[-2:]   # 2013→'13', ..., 2023→'23'

# 投影用的等面积CRS（中国Albers，用于准确计算路段长度）
CHINA_ALBERS = (
    '+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 '
    '+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
)

def compute_road_density(folder, h, w, dst_transform, dst_bounds,
                          road_base_dir, cell_area_m2):
    """
    路网线要素 → 像元级路网密度（m/m²）
    步骤：过滤道路类型 → 裁剪北京范围 → 投影计算长度 →
          像素分配（沿线采样） → 除以像元面积
    """
    shp_path = os.path.join(road_base_dir, folder, 'gis_osm_roads_free_1.shp')
    if not os.path.exists(shp_path):
        return None

    roads = gpd.read_file(shp_path)

    # 过滤道路类型
    if 'fclass' in roads.columns:
        roads = roads[roads['fclass'].isin(ROAD_TYPES_KEEP)].copy()
    if len(roads) == 0:
        return np.zeros((h, w), dtype=np.float32)

    # 裁剪到北京范围（WGS84下粗滤）
    from shapely.geometry import box
    bbox_poly = box(dst_bounds.left  - 0.05, dst_bounds.bottom - 0.05,
                    dst_bounds.right + 0.05, dst_bounds.top    + 0.05)
    roads = roads[roads.geometry.intersects(bbox_poly)].copy()
    if len(roads) == 0:
        return np.zeros((h, w), dtype=np.float32)

    # 投影到等面积坐标系计算长度
    roads_proj = roads.to_crs(CHINA_ALBERS)

    density = np.zeros((h, w), dtype=np.float32)
    inv_transform = ~dst_transform

    # 逐条路段采样并分配到像元
    SAMPLE_INTERVAL_M = 15.0   # 15m一个采样点（小于像元尺寸的一半）

    for geom_proj, geom_wgs in zip(roads_proj.geometry, roads.geometry):
        if geom_proj is None or geom_proj.is_empty:
            continue
        length_m = geom_proj.length
        if length_m < 1.0:
            continue

        n_pts = max(2, int(length_m / SAMPLE_INTERVAL_M))
        segment_len = length_m / n_pts   # 每个采样点代表的长度

        for i in range(n_pts):
            # 在WGS84几何上插值（保持坐标系一致）
            frac = (i + 0.5) / n_pts
            pt = geom_wgs.interpolate(frac, normalized=True)
            col_f = (pt.x - dst_transform.c) / dst_transform.a
            row_f = (pt.y - dst_transform.f) / dst_transform.e
            ci, ri = int(col_f + 0.5), int(row_f + 0.5)
            if 0 <= ri < h and 0 <= ci < w:
                density[ri, ci] += segment_len

    # 转换为密度（m/m²）
    density /= cell_area_m2
    return density


road_arrays = {}
computed_folders = {}   # folder → array，避免重复计算

for year in STUDY_YEARS:
    folder = year_to_folder(year)

    if folder in computed_folders:
        road_arrays[year] = computed_folders[folder]
        print(f"  {year}: 复用 folder={folder}/")
        continue

    print(f"  {year} (folder={folder}/)...", end=' ', flush=True)
    try:
        arr = compute_road_density(
            folder, h, w, dst_transform, dst_bounds,
            ROAD_BASE_DIR, cell_area_m2
        )
        if arr is not None:
            road_arrays[year] = arr
            computed_folders[folder] = arr
            v = arr[valid_mask & (arr > 0)]
            print(f"✓  有路像元={v.size:,}  max={v.max():.6f} m/m²")
        else:
            print(f"✗ SHP不存在")
    except Exception as e:
        print(f"✗ {e}")

if road_arrays:
    stack = np.stack(list(road_arrays.values()), axis=0)
    road_mean = np.nanmean(stack, axis=0).astype(np.float32)
    road_norm = normalize_01(np.log1p(road_mean), valid_mask)
    save_tif(road_norm, os.path.join(OUTPUT_DIR, 'road_density_30m.tif'),
             out_profile, valid_mask)
else:
    print("  ❌ 无路网数据！")


# ============================================================
# 五、积水点位 → 历史核密度
# ============================================================
print("\n[4/4] 历史积水点核密度...")

def parse_waterlog_file(path, file_year):
    """
    读取单年积水点Excel，统一处理5种时间格式，
    年份强制以file_year为准（修复年份错填问题）
    返回 DataFrame with [lon, lat, depth_cm, year]
    """
    xl     = pd.ExcelFile(path)
    df     = pd.read_excel(path, sheet_name=xl.sheet_names[0])

    # ── 识别经纬度列 ──────────────────────────────────
    lon_col = lat_col = dep_col = None
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ['longitude', '经度']:
            lon_col = col
        elif cl in ['latitude', '纬度']:
            lat_col = col
        elif 'depth_cm' in cl or cl == 'depth_cm':
            dep_col = col

    if lon_col is None or lat_col is None:
        raise ValueError(f"找不到经纬度列: {list(df.columns)}")

    df['lon']   = pd.to_numeric(df[lon_col], errors='coerce')
    df['lat']   = pd.to_numeric(df[lat_col], errors='coerce')
    df['depth'] = pd.to_numeric(df[dep_col], errors='coerce').fillna(30.0) \
                  if dep_col else 30.0
    df = df.dropna(subset=['lon', 'lat'])

    # ── 北京范围过滤 ──────────────────────────────────
    df = df[(df['lon'] >= 115.4) & (df['lon'] <= 117.6) &
            (df['lat'] >= 39.4)  & (df['lat'] <= 41.1)]

    df['year'] = file_year
    return df[['lon', 'lat', 'depth', 'year']]


def handle_2022_excel_serial(val):
    """
    2022文件时间列是Excel日期序列号（如44769）
    不需要解析日期，只是记录该文件year=2022即可
    此函数仅用于说明，实际处理已在parse_waterlog_file中完成
    """
    pass


all_wl = []

if os.path.exists(WATERLOG_DIR):
    for fname in sorted(os.listdir(WATERLOG_DIR)):
        # 跳过临时文件（~$开头）和非Excel文件
        if fname.startswith('~$') or fname.startswith('.'):
            continue
        if not (fname.endswith('.xlsx') or fname.endswith('.xls')):
            continue

        # 从文件名提取年份
        digits = re.findall(r'\d+', fname)
        if not digits:
            continue
        year_str = next((d for d in digits if len(d) == 4),
                         max(digits, key=len))
        try:
            file_year = int(year_str)
        except ValueError:
            continue

        # 研究期外可以用于密度估计，但2025也纳入
        if file_year < 2018 or file_year > 2025:
            continue

        path = os.path.join(WATERLOG_DIR, fname)
        print(f"  {fname} (year={file_year})...", end=' ', flush=True)
        try:
            df_yr = parse_waterlog_file(path, file_year)
            all_wl.append(df_yr)
            print(f"✓ {len(df_yr)} 个点")
        except Exception as e:
            print(f"✗ {e}")

    if all_wl:
        df_all = pd.concat(all_wl, ignore_index=True)
        print(f"\n  汇总: {len(df_all):,} 个积水点  "
              f"年份={sorted(df_all['year'].unique())}")

        # 积水深度作为权重（cm，合理裁剪后归一化）
        df_all['weight'] = np.clip(df_all['depth'], 5, 300) / 100.0

        density_wl = poi_to_density(
            df_all.rename(columns={'weight': 'w_col'}),
            h, w, dst_transform,
            bandwidth_deg=0.005,   # ≈500m，积水点属于精确位置
            weight_col='w_col'
        )
        wl_norm = normalize_01(density_wl, valid_mask)
        save_tif(wl_norm,
                 os.path.join(OUTPUT_DIR, 'waterlogging_point_density_30m.tif'),
                 out_profile, valid_mask)

        # 保存汇总点表（供Step4分析）
        df_all.to_csv(os.path.join(OUTPUT_DIR, 'waterlog_all_points.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"  ✅ waterlog_all_points.csv 已保存（{len(df_all):,} 条）")
    else:
        print("  ❌ 未成功读取任何积水点！")
else:
    print(f"  ❌ 目录不存在: {WATERLOG_DIR}")


# ============================================================
# 六、汇总验证
# ============================================================
print("\n" + "=" * 70)
print("Step 2.5b 完成！输出文件验证：")
print("=" * 70)

expected = [
    'hospital_density_30m.tif',
    'shelter_density_30m.tif',
    'firestation_density_30m.tif',
    'population_density_30m.tif',
    'road_density_30m.tif',
    'waterlogging_point_density_30m.tif',
]

ok_count = 0
for fname in expected:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        with rasterio.open(fpath) as src:
            arr = src.read(1)
        v = arr[arr != -9999.0]
        print(f"  ✅ {fname}")
        print(f"       有效={v.size:,}  均值={v.mean():.4f}  "
              f"范围=[{v.min():.4f},{v.max():.4f}]")
        ok_count += 1
    else:
        print(f"  ❌ 缺失: {fname}")

print(f"\n  {ok_count}/{len(expected)} 个文件生成成功")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"\n  ⬇️  下一步: 运行 step3_vsc_vulnerability.py")