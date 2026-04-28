"""
土壤参数文件探查脚本
读取三个已裁剪重采样的TIF文件，输出数据结构和统计信息
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FILES = {
    'Ks (饱和导水率, mm/h)':     r'F:\Data\src\fac\KSCH\ksch\Extract_K_SC1_Resample1.tif',
    'θs (饱和含水量, m³/m³)':    r'F:\Data\src\fac\THSCH\thsch\Extract_THSC1_Resample1.tif',
    'Psi (饱和毛管势, cm)':       r'F:\Data\src\fac\PSI\psi\Extract_PSI_1_Resample1.tif',
}

DEM_PATH = r'F:\Data\src\DEM数据\北京市_DEM_30m分辨率_NASA数据.tif'

print("=" * 70)
print("土壤参数 TIF 文件探查")
print("=" * 70)

# 读 DEM 获取参考空间信息
with rasterio.open(DEM_PATH) as dem:
    dem_crs       = dem.crs
    dem_transform = dem.transform
    dem_shape     = (dem.height, dem.width)
    dem_bounds    = dem.bounds
    dem_res       = dem.res

print(f"\n[DEM 参考信息]")
print(f"  CRS:       {dem_crs}")
print(f"  Shape:     {dem_shape[0]} 行 × {dem_shape[1]} 列")
print(f"  分辨率:    {dem_res}")
print(f"  Bounds:    {dem_bounds}")

results = {}

for label, fpath in FILES.items():
    print(f"\n{'='*60}")
    print(f"[探查] {label}")
    print(f"  路径: {fpath}")

    if not os.path.exists(fpath):
        print(f"  ❌ 文件不存在！")
        results[label] = None
        continue

    with rasterio.open(fpath) as src:
        crs       = src.crs
        transform = src.transform
        shape     = (src.height, src.width)
        bounds    = src.bounds
        res       = src.res
        nodata    = src.nodata
        dtype     = src.dtypes[0]
        count     = src.count

        data = src.read(1).astype(np.float64)

    print(f"  CRS:       {crs}")
    print(f"  Shape:     {shape[0]} 行 × {shape[1]} 列")
    print(f"  分辨率:    {res}")
    print(f"  Bounds:    {bounds}")
    print(f"  NoData:    {nodata}")
    print(f"  数据类型:  {dtype}  波段数: {count}")

    # 空间对齐检查
    shape_match  = (shape == dem_shape)
    res_match    = all(abs(a - b) < 1e-8 for a, b in zip(res, dem_res))
    bounds_match = all(
        abs(a - b) < 1e-4
        for a, b in zip(
            [bounds.left, bounds.bottom, bounds.right, bounds.top],
            [dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top]
        )
    )
    print(f"\n  对齐检查:")
    print(f"    Shape一致: {'✓' if shape_match else '✗ 不一致！'}")
    print(f"    分辨率一致: {'✓' if res_match else '✗ 不一致！'}")
    print(f"    Bounds一致: {'✓' if bounds_match else '✗ 不一致（可接受小偏差）'}")

    # 处理 nodata
    if nodata is not None:
        invalid_mask = (data == nodata) | np.isinf(data)
    else:
        # 尝试检测常见 nodata 值
        invalid_mask = (data < -1e30) | np.isinf(data)
        if invalid_mask.any():
            print(f"  ⚠️  未声明 nodata，检测到极大负值: min={data.min():.4f}")

    data_clean = data.copy()
    data_clean[invalid_mask] = np.nan

    valid = data_clean[~np.isnan(data_clean)]
    n_valid   = valid.size
    n_invalid = invalid_mask.sum()
    total     = data.size

    print(f"\n  像元统计:")
    print(f"    总像元:   {total:,}")
    print(f"    有效像元: {n_valid:,}  ({n_valid/total*100:.2f}%)")
    print(f"    无效像元: {n_invalid:,}  ({n_invalid/total*100:.2f}%)")

    if n_valid > 0:
        print(f"\n  数值统计 (有效像元):")
        print(f"    Min:    {valid.min():.4f}")
        print(f"    Max:    {valid.max():.4f}")
        print(f"    Mean:   {valid.mean():.4f}")
        print(f"    Std:    {valid.std():.4f}")
        print(f"    Median: {np.median(valid):.4f}")
        print(f"    P1:     {np.percentile(valid, 1):.4f}")
        print(f"    P5:     {np.percentile(valid, 5):.4f}")
        print(f"    P25:    {np.percentile(valid, 25):.4f}")
        print(f"    P75:    {np.percentile(valid, 75):.4f}")
        print(f"    P95:    {np.percentile(valid, 95):.4f}")
        print(f"    P99:    {np.percentile(valid, 99):.4f}")

        # 诊断可疑值
        neg_count = (valid < 0).sum()
        zero_count = (valid == 0).sum()
        print(f"\n  诊断:")
        print(f"    负值像元: {neg_count:,}  ({neg_count/n_valid*100:.2f}%)")
        print(f"    零值像元: {zero_count:,}  ({zero_count/n_valid*100:.2f}%)")
        if neg_count > 0:
            neg_vals = valid[valid < 0]
            print(f"    负值范围: [{neg_vals.min():.4f}, {neg_vals.max():.4f}]")

    results[label] = {'data': data_clean, 'valid': n_valid, 'shape': shape}

# ============================================================
# 可视化：三个参数空间分布 + 直方图
# ============================================================
print("\n\n生成可视化...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('土壤参数 TIF 文件探查结果', fontsize=14, fontweight='bold')

for col, (label, fpath) in enumerate(FILES.items()):
    if results[label] is None:
        continue
    data_c = results[label]['data']
    valid  = data_c[~np.isnan(data_c)]

    # 上行：空间分布图
    ax_map = axes[0, col]
    clip_lo = np.percentile(valid, 2) if valid.size > 0 else 0
    clip_hi = np.percentile(valid, 98) if valid.size > 0 else 1
    im = ax_map.imshow(np.clip(data_c, clip_lo, clip_hi),
                       cmap='viridis', interpolation='nearest')
    ax_map.set_title(f'{label}\n空间分布', fontsize=10, fontweight='bold')
    ax_map.axis('off')
    plt.colorbar(im, ax=ax_map, shrink=0.8)
    ax_map.text(0.02, 0.02,
                f"Mean={np.nanmean(data_c):.4f}\nStd={np.nanstd(data_c):.4f}",
                transform=ax_map.transAxes, fontsize=8.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 下行：直方图
    ax_hist = axes[1, col]
    ax_hist.hist(valid, bins=80, color='steelblue', alpha=0.75, edgecolor='white', linewidth=0.3)
    ax_hist.axvline(np.mean(valid), color='red', linewidth=1.5, linestyle='--', label=f"均值={np.mean(valid):.4f}")
    ax_hist.axvline(np.median(valid), color='orange', linewidth=1.5, linestyle='-', label=f"中位={np.median(valid):.4f}")
    ax_hist.set_title(f'{label}\n直方图', fontsize=10, fontweight='bold')
    ax_hist.set_xlabel('数值')
    ax_hist.set_ylabel('频数')
    ax_hist.legend(fontsize=8)
    ax_hist.text(0.98, 0.95,
                 f"Min={valid.min():.4f}\nMax={valid.max():.4f}\nN={valid.size:,}",
                 transform=ax_hist.transAxes, fontsize=8, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

plt.tight_layout()
out_vis = './soil_params_inspection.png'
plt.savefig(out_vis, dpi=180, bbox_inches='tight')
plt.close()
print(f"✅ 可视化已保存: {out_vis}")

# ============================================================
# 对齐总结
# ============================================================
print("\n" + "=" * 70)
print("总结：是否与DEM完全对齐？")
print("=" * 70)
for label, fpath in FILES.items():
    r = results[label]
    if r is None:
        print(f"  {label}: ❌ 文件不存在")
    elif r['shape'] == dem_shape:
        print(f"  {label}: ✓ 行列数与DEM一致 ({r['shape'][0]}×{r['shape'][1]})")
    else:
        print(f"  {label}: ⚠️  行列数不同 ({r['shape'][0]}×{r['shape'][1]} vs DEM {dem_shape[0]}×{dem_shape[1]})")
        print(f"         → Step1将使用 out_shape=(h,w) 重采样读取")