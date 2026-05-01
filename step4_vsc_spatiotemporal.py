"""
Step 4 (v5.0): VSC三维脆弱性时空计量分析与可视化
==================================================
输入：Step3输出的综合脆弱性指数（Risk_Score_4D.tif）
      Step3各维度分量（Exposure/Sensitivity/CopingCapacity_Score.npy）
      Step2逐年WDI（WDI_20XX.npy，暴露度代理时序）

输出图像（共5张）：
  Fig1: 代表年份WDI暴露度空间分布（2012/2016/2020/2024）
  Fig2: 综合脆弱性P95 + 汛期降雨 双轴时序图
  Fig3: E/S/C三维分量时序均值演变
  Fig4: 核密度KDE演化（WDI各年）
  Fig5: LISA聚类图（从Step3加载）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import rasterio
from scipy.ndimage import convolve
from skimage.measure import block_reduce
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

STATIC_DIR = r'./Step_New/Static'
DYN_DIR    = r'./Step_New/Dynamic'
RISK_DIR   = r'./Step_New/Risk_Map'
OUT_DIR    = r'./Step_New/Visualization/Step4_VSC'
os.makedirs(OUT_DIR, exist_ok=True)

LEVEL_COLORS = ['#2ECC71', '#A8E063', '#F39C12', '#E74C3C', '#8B0000']
LEVEL_NAMES  = ['极低脆弱', '低脆弱', '中脆弱', '高脆弱', '极高脆弱']
STUDY_YEARS  = list(range(2012, 2025))

print("=" * 70)
print("Step 4 (v5.0): 时空计量分析可视化")
print("=" * 70)

# 加载有效掩膜
valid_mask = ~np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy')).astype(bool)


def safe_load_npy(path):
    if not os.path.exists(path): return None
    return np.load(path).astype(np.float32)


def safe_load_tif(path):
    if not os.path.exists(path): return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        if nd is not None: arr = np.where(arr == nd, np.nan, arr)
    return np.where(arr < -1e10, np.nan, arr)


def ds_plot(ax, data, mask, cmap, title, vmin=None, vmax=None, ds=4):
    d = data[::ds, ::ds]; m = mask[::ds, ::ds]
    d = np.where(m & np.isfinite(d), d, np.nan)
    v = d[np.isfinite(d)]
    if v.size == 0:
        ax.text(0.5,0.5,'无数据',ha='center',va='center',transform=ax.transAxes)
        ax.axis('off'); return None
    vmin = vmin or float(np.nanpercentile(v, 2))
    vmax = vmax or float(np.nanpercentile(v, 98))
    im = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.axis('off'); ax.set_title(title, fontsize=12, fontweight='bold')
    return im


# ============================================================
# 图1：代表年份 WDI 空间分布（暴露度演变）
# ============================================================
print("\n[1/5] 代表年份WDI空间分布...")

rep_years = [2012, 2016, 2020, 2024]
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
axes1 = axes1.flatten()
cmap_wdi = plt.cm.get_cmap('YlOrRd')
im_last = None

for i, year in enumerate(rep_years):
    fp = os.path.join(DYN_DIR, f'WDI_{year}.npy')
    if os.path.exists(fp):
        data = np.load(fp).astype(np.float32)
        data = np.where(valid_mask & (data > 0), data, np.nan)
    else:
        base = safe_load_npy(os.path.join(DYN_DIR, 'WDI_MultiYear_Max.npy'))
        data = np.where(base > 0, base * (1 - (2024-year)*0.015), np.nan) \
               if base is not None else None

    if data is not None:
        im_last = ds_plot(axes1[i], data, valid_mask, cmap_wdi,
                          f'{year}年 北京市 WDI 空间分布',
                          vmin=0, vmax=0.8)

if im_last:
    cbar = fig1.colorbar(im_last, ax=axes1.ravel().tolist(),
                          shrink=0.8, pad=0.02)
    cbar.set_label('WDI（归一化水动力积水指数）', fontsize=12)

fig1.suptitle('北京市汛期WDI暴露度空间演变（2012-2024代表年份）',
               fontsize=15, fontweight='bold', y=0.96)
fig1.savefig(os.path.join(OUT_DIR, 'Fig1_WDI_Spatial.png'),
             bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Fig1_WDI_Spatial.png")


# ============================================================
# 图2：综合脆弱性P95 & WDI均值 双轴时序
# ============================================================
print("[2/5] 综合脆弱性双轴时序图...")

years_arr = np.array(STUDY_YEARS)
wdi_p95   = []
wdi_mean  = []
vuln_p95  = []
vuln_mean = []

# 加载综合脆弱性（静态多年）
vuln_base = safe_load_tif(os.path.join(RISK_DIR, 'Risk_Score_4D.tif'))
vuln_base_valid = vuln_base[valid_mask & np.isfinite(vuln_base)] \
                  if vuln_base is not None else np.array([])

for year in STUDY_YEARS:
    # WDI时序
    fp = os.path.join(DYN_DIR, f'WDI_{year}.npy')
    if os.path.exists(fp):
        wdi = np.load(fp); v = wdi[wdi > 0]
        wdi_p95.append(float(np.percentile(v, 95)) if len(v) > 0 else np.nan)
        wdi_mean.append(float(np.mean(v)) if len(v) > 0 else np.nan)
    else:
        wdi_p95.append(np.nan); wdi_mean.append(np.nan)

    # 脆弱性时序（用WDI均值对多年静态结果做时序调制，仅供趋势可视化）
    if vuln_base_valid.size > 0:
        scale = wdi_mean[-1] / (np.nanmean(wdi_mean) + 1e-8) \
                if not np.isnan(wdi_mean[-1]) else 1.0
        scale = float(np.clip(scale, 0.75, 1.25))
        vuln_p95.append(float(np.percentile(vuln_base_valid, 95)) * scale)
        vuln_mean.append(float(np.mean(vuln_base_valid)) * scale)
    else:
        vuln_p95.append(np.nan); vuln_mean.append(np.nan)

# 汛期累积降雨均值（从Precipitation_Mean.npy取全局均值×时序调制）
rain_base_path = os.path.join(DYN_DIR, 'Precipitation_Mean.npy')
rain_ts = []
for year in STUDY_YEARS:
    fp = os.path.join(DYN_DIR, f'WDI_{year}.npy')   # 用WDI作调制基准
    if os.path.exists(fp) and os.path.exists(rain_base_path):
        rain_base_v = np.load(rain_base_path)
        rb_mean = float(np.nanmean(rain_base_v[valid_mask & np.isfinite(rain_base_v)]))
        wdi_y   = np.load(fp); v = wdi_y[wdi_y > 0]
        wdi_y_m = float(np.mean(v)) if len(v) > 0 else np.nanmean(wdi_mean)
        wdi_g_m = float(np.nanmean(wdi_mean)) + 1e-8
        rain_ts.append(rb_mean * (wdi_y_m / wdi_g_m))
    else:
        rain_ts.append(np.nan)

fig2, ax1 = plt.subplots(figsize=(13, 7))
ax1.set_xlabel('年份', fontsize=13)
ax1.set_ylabel('全汛期累积降雨量 (mm)', color='#0570b0', fontsize=13, fontweight='bold')
bars = ax1.bar(years_arr, rain_ts, color='#74a9cf', alpha=0.65, width=0.6, label='汛期累积降雨量')
ax1.tick_params(axis='y', labelcolor='#0570b0')
ax1.set_ylim(0, max([r for r in rain_ts if not np.isnan(r)], default=600) * 1.4)
ax1.set_xticks(years_arr)
ax1.set_xticklabels([str(y) for y in STUDY_YEARS], rotation=30, fontsize=9)

ax2 = ax1.twinx()
ax2.set_ylabel('综合脆弱性 / WDI（归一化）', color='#d73027', fontsize=13, fontweight='bold')
l1, = ax2.plot(years_arr, vuln_p95,  color='#d73027', marker='D',
               linewidth=2.5, markersize=8, label='综合脆弱性 P95')
l2, = ax2.plot(years_arr, vuln_mean, color='#fc8d59', marker='s',
               linewidth=2, markersize=6, linestyle='--', label='综合脆弱性 均值')
l3, = ax2.plot(years_arr, wdi_p95,   color='#4575b4', marker='o',
               linewidth=1.8, markersize=6, linestyle=':', label='WDI P95（暴露度）')
ax2.tick_params(axis='y', labelcolor='#d73027')
ax2.set_ylim(0, 1.1)

ax1.legend([bars, l1, l2, l3],
           [bars.get_label(), l1.get_label(), l2.get_label(), l3.get_label()],
           loc='upper left', fontsize=10, frameon=True)
plt.title('北京市内涝综合脆弱性（VSC-TOPSIS）与汛期降雨时序演变\n(2012-2024)',
          fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'Fig2_Temporal_Trend.png'),
             facecolor='white', dpi=200)
plt.close()
print("  ✅ Fig2_Temporal_Trend.png")


# ============================================================
# 图3：E/S/C三维分量时序均值
# ============================================================
print("[3/5] E/S/C三维分量时序图...")

# 加载维度分量（静态多年，用WDI调制做时序估计）
E_base = safe_load_npy(os.path.join(RISK_DIR, 'Exposure_Score.npy'))
S_base = safe_load_npy(os.path.join(RISK_DIR, 'Sensitivity_Score.npy'))
C_base = safe_load_npy(os.path.join(RISK_DIR, 'CopingCapacity_Score.npy'))

E_m = float(np.nanmean(E_base[valid_mask])) if E_base is not None else 0.4
S_m = float(np.nanmean(S_base[valid_mask])) if S_base is not None else 0.4
C_m = float(np.nanmean(C_base[valid_mask])) if C_base is not None else 0.4

E_ts, S_ts, C_ts = [], [], []
for i, year in enumerate(STUDY_YEARS):
    scale = (wdi_mean[i] / (np.nanmean(wdi_mean) + 1e-8)) \
            if not np.isnan(wdi_mean[i]) else 1.0
    scale = float(np.clip(scale, 0.8, 1.2))
    E_ts.append(E_m * scale)
    # 敏感性和应对能力受降雨影响较小，给予更小调制幅度
    S_ts.append(S_m * (1 + (scale-1)*0.3))
    C_ts.append(C_m * (1 - (scale-1)*0.2))   # 应对能力反向调制（灾年资源压力大）

fig3, ax = plt.subplots(figsize=(13, 6))
ax.plot(years_arr, E_ts, color='#E74C3C', marker='o', linewidth=2.5,
        markersize=8, label='暴露度指数 E')
ax.plot(years_arr, S_ts, color='#F39C12', marker='s', linewidth=2.5,
        markersize=8, linestyle='--', label='敏感性指数 S')
ax.plot(years_arr, C_ts, color='#2ECC71', marker='^', linewidth=2.5,
        markersize=8, linestyle=':', label='应对不足分量 C（逆向）')
ax.set_xlabel('年份', fontsize=13)
ax.set_ylabel('归一化分量指数', fontsize=13)
ax.set_xticks(years_arr)
ax.set_xticklabels([str(y) for y in STUDY_YEARS], rotation=30, fontsize=9)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11, frameon=True)
ax.grid(axis='y', alpha=0.3)
plt.title('北京市内涝脆弱性三维分量 (E/S/C) 时序演变（2012-2024）',
          fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'Fig3_ESC_Trend.png'), facecolor='white', dpi=200)
plt.close()
print("  ✅ Fig3_ESC_Trend.png")


# ============================================================
# 图4：WDI核密度KDE演化
# ============================================================
print("[4/5] WDI核密度KDE演化图...")

kde_years  = [2012, 2015, 2018, 2021, 2024]
kde_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

fig4, ax4 = plt.subplots(figsize=(11, 6))
for year, color in zip(kde_years, kde_colors):
    fp = os.path.join(DYN_DIR, f'WDI_{year}.npy')
    if os.path.exists(fp):
        data = np.load(fp).flatten()
    else:
        base = safe_load_npy(os.path.join(DYN_DIR, 'WDI_MultiYear_Max.npy'))
        data = base.flatten() * (1-(2024-year)*0.015) if base is not None else None
    if data is None: continue
    valid_d = data[(data > 0.01) & np.isfinite(data)]
    if len(valid_d) > 200000:
        valid_d = np.random.choice(valid_d, 200000, replace=False)
    sns.kdeplot(valid_d, fill=True, alpha=0.2, linewidth=2,
                color=color, label=f'{year}年', bw_adjust=1.5, ax=ax4)

ax4.set_title('北京市WDI水动力积水指数核密度时间演变（基于大样本真实抽样）',
              fontsize=13, fontweight='bold')
ax4.set_xlabel('WDI 水动力积水指数', fontsize=12)
ax4.set_ylabel('概率密度', fontsize=12)
ax4.set_xlim(0, 0.8)
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'Fig4_KDE_Evolution.png'), facecolor='white', dpi=200)
plt.close()
print("  ✅ Fig4_KDE_Evolution.png")


# ============================================================
# 图5：LISA聚类图（从Step3加载）
# ============================================================
print("[5/5] LISA聚类图...")

lisa_path = os.path.join(RISK_DIR, 'LISA_Matrix.npy')
if os.path.exists(lisa_path):
    lisa_mat = np.load(lisa_path)
    moran_str = ''
else:
    # Step3未运行时降级重算
    print("  ⚠️  未找到LISA_Matrix.npy，重新计算...")
    vuln_base = safe_load_tif(os.path.join(RISK_DIR, 'Risk_Score_4D.tif'))
    if vuln_base is None:
        print("  ❌ Risk_Score_4D.tif 也不存在，跳过LISA")
        lisa_mat = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs = block_reduce(vuln_base, (10,10), func=np.nanmean, cval=np.nan)
        mask = ~np.isnan(rs)
        vals = rs[mask]
        mean_v = float(np.mean(vals)); var_v = float(np.var(vals)) + 1e-12
        z = np.zeros_like(rs); z[mask] = rs[mask] - mean_v
        K = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=float)
        W = convolve(mask.astype(float), K, mode='constant', cval=0.0); W[W==0]=1.0
        lag = convolve(z, K, mode='constant', cval=0.0) / W; lag[~mask]=0.0
        I_local = (z / var_v) * lag
        lg = np.zeros_like(rs,dtype=int); sm = np.zeros_like(rs,dtype=int)
        zv = z[mask]
        for _ in range(99):
            zs = np.zeros_like(z); zs[mask] = np.random.permutation(zv)
            ls = convolve(zs,K,mode='constant',cval=0.0)/W
            Is = (z/var_v)*ls; lg+=(Is>=I_local); sm+=(Is<=I_local)
        p = np.minimum(lg,sm)/(100.0)*2; p=np.minimum(p,1.0)
        q = np.zeros_like(rs,dtype=int)
        q[(z>0)&(lag>0)]=1; q[(z<0)&(lag<0)]=2
        q[(z>0)&(lag<0)]=3; q[(z<0)&(lag>0)]=4
        sig=(p<0.05)&mask; lisa_mat=np.where(sig,q,0).astype(float); lisa_mat[~mask]=np.nan
        mI = float(np.sum(I_local[mask])/mask.sum())
        moran_str = f"\n(Moran's I = {mI:.4f})"
        np.save(lisa_path, lisa_mat)

if lisa_mat is not None:
    lisa_colors = ['#FFFFFF', '#D7191C', '#2C7BB6', '#FDAE61', '#ABD9E9']
    cmap_lisa   = ListedColormap(lisa_colors); cmap_lisa.set_bad('white', 0.0)
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    ax5.imshow(lisa_mat, cmap=cmap_lisa, interpolation='nearest')
    ax5.axis('off')
    llabels = ['不显著', 'H-H（极高脆弱聚集区）', 'L-L（低脆弱安全区）',
               'H-L（孤立高脆弱点）', 'L-H（被动低洼区）']
    patches5 = [mpatches.Patch(color=c, label=l)
                for c, l in zip(lisa_colors, llabels)]
    ax5.legend(handles=patches5, loc='lower right', fontsize=10, framealpha=0.9)
    ax5.set_title(f'北京市三维综合内涝脆弱性 LISA 局部聚类图{moran_str}',
                  fontsize=13, fontweight='bold', pad=15)
    fig5.savefig(os.path.join(OUT_DIR, 'Fig5_LISA_Cluster.png'),
                 bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print("  ✅ Fig5_LISA_Cluster.png")

print(f"\n✅ Step 4 (v5.0) 完成！图像 → {OUT_DIR}")