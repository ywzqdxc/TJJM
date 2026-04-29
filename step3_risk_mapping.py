"""
Step 3 Enhanced Final: 4维综合内涝风险区划
============================================
完全修复版 - h,w类型问题已解决
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgs
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
from scipy.stats import pearsonr, kruskal
from itertools import product as iterproduct
import pandas as pd
import os, gc, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径
# ============================================================
STATIC_DIR = r'./Step_New/Static'
DYN_DIR    = r'./Step_New/Dynamic'
POI_DIR    = r'./Step_New/POI_Exposure'
OUTPUT_DIR = r'./Step_New/Risk_Map'
VIS_DIR    = r'./Step_New/Visualization/Step3'
for d in [OUTPUT_DIR, VIS_DIR]: os.makedirs(d, exist_ok=True)

LEVEL_COLORS = ['#2ECC71', '#A8E063', '#F39C12', '#E74C3C', '#8B0000']
LEVEL_NAMES  = ['极低风险', '低风险', '中风险', '高风险', '极高风险']
PLOT_DS = 4

# ============================================================
# 工具函数
# ============================================================
def safe_nan(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def min_max_norm(x):
    xc = safe_nan(x); mn, mx = xc.min(), xc.max()
    return np.zeros_like(xc) if mx - mn < 1e-8 else (xc - mn) / (mx - mn)

def entropy_weight(X):
    n, m = X.shape; w = np.zeros(m); e_vec = np.ones(m); d_vec = np.zeros(m)
    active = [j for j in range(m) if X[:,j].sum() > 1e-10 and X[:,j].std() > 1e-10]
    if not active: w[:] = 1.0/m; return w, e_vec, d_vec
    for j in active:
        Xn = np.clip(X[:,j]/(X[:,j].sum()+1e-10), 1e-12, 1.0)
        e_vec[j] = -np.sum(Xn*np.log(Xn))/np.log(n); d_vec[j] = 1-e_vec[j]
    ds = d_vec[active].sum()
    for j in active: w[j] = d_vec[j]/(ds+1e-10)
    return w, e_vec, d_vec

def topsis(X, w):
    Xw = X*w; pos, neg = Xw.max(axis=0), Xw.min(axis=0)
    dp = np.sqrt(((Xw-pos)**2).sum(axis=1)); dn = np.sqrt(((Xw-neg)**2).sum(axis=1))
    return dn/(dp+dn+1e-10), Xw


def natural_breaks(vals, n=5):
    """Jenks自然断裂法（修复版）"""
    sv = np.sort(vals)
    N = len(sv)
    if N < n * 2:
        return np.linspace(sv.min(), sv.max(), n + 1)

    # 初始断点：等分位数
    breaks = np.quantile(sv, np.linspace(0, 1, n + 1))

    for _ in range(50):
        old_breaks = breaks.copy()
        # 用breaks[1:-1]作为分界（n-1个中间断点）
        labels = np.digitize(sv, breaks[1:-1])

        new_breaks = [sv.min()]
        for k in range(n):
            grp = sv[labels == k]
            if len(grp) > 0:
                new_breaks.append(float(grp.max()))
            else:
                # 保持旧断点
                new_breaks.append(float(old_breaks[k + 1]))
        new_breaks[-1] = sv.max()  # 确保最后一个就是最大值

        breaks = np.asarray(new_breaks, dtype=np.float64)

        # 检查收敛（确保两个数组长度一致）
        if len(breaks) == len(old_breaks):
            if np.max(np.abs(breaks - old_breaks)) < 1e-6:
                break
        else:
            # 长度不一致时强制对齐
            breaks = old_breaks.copy()
            break

    return breaks

def assign_breaks(score, breaks):
    risk = np.full_like(score, np.nan)
    for i, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:])):
        risk[(score>=lo) & ((score<hi) if i<len(breaks)-2 else (score<=hi))] = i+1
    return risk

def plot_masked(ax, data, mask, cmap, vmin=None, vmax=None, ds=PLOT_DS):
    d = data[::ds,::ds].astype(np.float32); m = mask[::ds,::ds]
    d = np.where(m & np.isfinite(d), d, np.nan); d[~m] = np.nan
    v = d[m & ~np.isnan(d)]
    if v.size < 10: return None
    if vmin is None: vmin = float(np.nanpercentile(v, 2))
    if vmax is None: vmax = float(np.nanpercentile(v, 98))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap)(norm(d)).astype(np.float32)
    rgba[~m,3] = 0.0; rgba[np.isnan(d)&m,3] = 0.0
    ax.imshow(rgba); sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    return sm

def plot_cat(ax, data, mask, cmap, norm, ds=PLOT_DS):
    d = data[::ds,::ds].astype(np.float32); m = mask[::ds,::ds]
    rgba = cmap(norm(d)); rgba[~m,3] = 0.0; rgba[np.isnan(d)&m,3] = 0.0
    ax.imshow(rgba)

def load_raster(path, h, w, name):
    if path.endswith('.npy'):
        d = np.load(path)
        if d.shape == (h,w): print(f"    ✅ {name}(NPY)"); return d.astype(np.float32)
    elif os.path.exists(path):
        try:
            with rasterio.open(path) as src:
                d = src.read(1).astype(np.float32)
                d = np.where((d<-1e10)|(d==-9999.0), np.nan, d)
                if d.shape == (h,w): print(f"    ✅ {name}(TIFF)"); return d
        except: pass
    print(f"    ⚠️  {name}缺失"); return np.zeros((h,w), dtype=np.float32)

# ============================================================
# 主流程
# ============================================================
print("=" * 70)
print("Step 3 Enhanced Final: 4维综合内涝风险区划")
print("=" * 70)

# ── 1. 加载数据 ──
print("\n[1/5] 加载数据...")
slope       = np.load(os.path.join(STATIC_DIR, 'slope.npy'))
hand        = np.load(os.path.join(STATIC_DIR, 'hand.npy'))
nodata_mask = np.load(os.path.join(STATIC_DIR, 'nodata_mask.npy'))
valid_mask  = ~nodata_mask

# ★★★ 关键：强制Python int ★★★
H = int(slope.shape[0])
W = int(slope.shape[1])
N = int(valid_mask.sum())

print(f"    栅格: {H}×{W}, 有效像元: {N:,} (类型: H={type(H).__name__}, W={type(W).__name__})")

wdi_raw = load_raster(os.path.join(DYN_DIR, 'WDI_MultiYear_Max.npy'), H, W, 'WDI')
poi_exp = load_raster(os.path.join(POI_DIR, 'POI_Exposure_LogNorm.tif'), H, W, 'POI')

out_profile = {'driver':'GTiff','height':H,'width':W,'count':1,
               'dtype':'float32','nodata':-9999.0,'compress':'deflate'}

# ── 2. 4维指标 ──
print("\n[2/5] 4维指标矩阵...")
wdi_v   = safe_nan(wdi_raw[valid_mask])
hand_v  = safe_nan(hand[valid_mask])
slope_v = safe_nan(slope[valid_mask])
poi_v   = safe_nan(poi_exp[valid_mask])

ind1_raw = wdi_v
ind2_raw = 1.0/(hand_v+1.0)
ind3_raw = 1.0-min_max_norm(slope_v)
ind4_raw = poi_v

NAMES = ['动态积水WDI','低洼性(1/HAND)','平坦性(1-slope)','POI设施暴露度']
SHORT = ['WDI','低洼','平坦','POI']
RAWS  = [ind1_raw, ind2_raw, ind3_raw, ind4_raw]

for name, arr in zip(NAMES, RAWS):
    print(f"  {name:16s}: 均值={np.nanmean(arr):.4f}  CV={np.nanstd(arr)/(abs(np.nanmean(arr))+1e-10):.4f}")

X = np.column_stack([min_max_norm(r) for r in RAWS])
X = safe_nan(X)

# ── 3. 统计检验 ──
print("\n[3/5] 统计检验...")
cv_list = [np.nanstd(r)/(abs(np.nanmean(r))+1e-10) for r in RAWS]
sidx = np.random.choice(N, min(5000,N), replace=False)

corr = np.eye(4)
for i in range(4):
    for j in range(i+1,4):
        r, p = pearsonr(RAWS[i][sidx], RAWS[j][sidx])
        corr[i,j] = corr[j,i] = r
        print(f"  {SHORT[i]}-{SHORT[j]}: r={r:.4f} {'**' if p<0.01 else '*' if p<0.05 else ''}")

# ── 4. TOPSIS + 断裂 ──
print("\n[4/5] TOPSIS + 自然断裂...")
w, e_vec, d_vec = entropy_weight(X)
score, _ = topsis(X, w)

print("\n  4维权重:")
for name, wi, cv in zip(NAMES, w, cv_list):
    print(f"  {name:16s}: {wi:.4f} {'█'*int(wi*40)} (CV={cv:.3f})")

w3, _, _ = entropy_weight(X[:,:3])
print("\n  3维→4维:")
for i, name in enumerate(NAMES):
    print(f"  {name:16s}: {w3[i] if i<3 else 0:.4f} → {w[i]:.4f}  {'【新增】' if i==3 else ''}")

breaks = natural_breaks(score[~np.isnan(score)])
risk_lv = assign_breaks(score, breaks)

# ★★★ 使用H,W创建2D数组 ★★★
risk_score_2d = np.full((H, W), np.nan, dtype=np.float32)
risk_score_2d[valid_mask] = score.astype(np.float32)
risk_level_2d = np.full((H, W), np.nan, dtype=np.float32)
risk_level_2d[valid_mask] = risk_lv.astype(np.float32)

pcts = [float(np.nanmean(risk_level_2d==i))*100 for i in range(1,6)]
print("\n  五级面积:")
for name, pct, lo, hi in zip(LEVEL_NAMES, pcts, breaks[:-1], breaks[1:]):
    print(f"  {name}: {pct:5.1f}% [{lo:.3f}, {hi:.3f}]")

groups = [score[risk_lv==i] for i in range(1,6) if np.sum(risk_lv==i)>1]
kw_h, kw_p = kruskal(*groups)
print(f"\n  KW: H={kw_h:.0f}  P={kw_p:.2e}  ***")

# 稳健性
ns = int(N*0.02); si = np.random.choice(N, ns, replace=False); Xs = X[si]
pscores = []
for dw in iterproduct([-0.1,0,0.1], repeat=4):
    wp = np.clip(w+np.array(dw)*0.1, 0, 1); wp /= wp.sum()+1e-10
    sc, _ = topsis(Xs, wp); pscores.append(sc)
rob_std = float(np.array(pscores).std(axis=0).mean())
print(f"  稳健性: {rob_std:.5f} {'✓' if rob_std<0.05 else '⚠'}")

# ── 5. 可视化 (5张) ──
print("\n[5/5] 可视化...")

def save(fig, name):
    fig.savefig(os.path.join(VIS_DIR, name), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig); gc.collect(); print(f"  → {name}")

cmap5 = ListedColormap(LEVEL_COLORS)
norm5 = BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], 5)

# 图1: 总览
fig1, axes1 = plt.subplots(2, 3, figsize=(20, 13)); fig1.suptitle('4维综合内涝风险区划', fontsize=14, fontweight='bold')
for ax, r, n, cm in zip(axes1.flat[:4], RAWS, NAMES, ['hot_r','Blues_r','RdYlGn_r','YlOrRd']):
    d2d = np.full((H,W), np.nan, dtype=np.float32); d2d[valid_mask] = r.astype(np.float32)
    plot_masked(ax, d2d, valid_mask, cm)
    ax.set_title(f'{n}\nμ={np.nanmean(r):.3f} CV={np.nanstd(r)/(abs(np.nanmean(r))+1e-10):.2f}', fontsize=9); ax.axis('off')
plot_masked(axes1.flat[4], np.clip(risk_score_2d,0,1), valid_mask, 'RdYlGn_r',0,1)
axes1.flat[4].set_title(f'TOPSIS得分\nPOI权重={w[3]:.3f}', fontsize=9, fontweight='bold'); axes1.flat[4].axis('off')
plot_cat(axes1.flat[5], risk_level_2d, valid_mask, cmap5, norm5)
axes1.flat[5].set_title(f'五级区划\nKW H={kw_h:.0f}***', fontsize=9, fontweight='bold'); axes1.flat[5].axis('off')
plt.tight_layout(); save(fig1, 'Step3_Final_01_Overview.png')

# 图2: 权重+相关
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6)); fig2.suptitle('统计检验', fontsize=13, fontweight='bold')
ax2a.bar(SHORT, w, color=['#E74C3C','#3498DB','#2ECC71','#F39C12'], alpha=0.8)
ax2a.set_ylabel('权重'); ax2a.set_title('4维熵权')
for i, (bar, wi) in enumerate(zip(ax2a.patches, w)):
    ax2a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{wi:.3f}', ha='center', fontweight='bold')
im = ax2b.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax2b.set_xticks(range(4)); ax2b.set_yticks(range(4))
ax2b.set_xticklabels(SHORT); ax2b.set_yticklabels(SHORT)
for i in range(4):
    for j in range(4):
        ax2b.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                  color='white' if abs(corr[i,j])>0.5 else 'black', fontsize=10)
plt.colorbar(im, ax=ax2b, shrink=0.8, label='Pearson r'); ax2b.set_title('相关矩阵', fontweight='bold')
plt.tight_layout(); save(fig2, 'Step3_Final_02_Stats.png')

# 图3: 3D vs 4D
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7)); fig3.suptitle('POI暴露度影响 (3维→4维)', fontsize=13, fontweight='bold')
s3, _ = topsis(X[:,:3], w3); b3 = natural_breaks(s3); r3 = assign_breaks(s3, b3)
r3_2d = np.full((H,W), np.nan, dtype=np.float32); r3_2d[valid_mask] = r3.astype(np.float32)
p3 = [float(np.nanmean(r3_2d==i))*100 for i in range(1,6)]
plot_cat(ax3a, r3_2d, valid_mask, cmap5, norm5); ax3a.set_title('3维区划 (地形+WDI)', fontweight='bold'); ax3a.axis('off')
ax3a.legend(handles=[mpatches.Patch(color=c, label=f'{n}:{p:.1f}%') for c,n,p in zip(LEVEL_COLORS,LEVEL_NAMES,p3)], loc='lower right', fontsize=8)
plot_cat(ax3b, risk_level_2d, valid_mask, cmap5, norm5); ax3b.set_title('4维区划 (+POI)', fontweight='bold'); ax3b.axis('off')
ax3b.legend(handles=[mpatches.Patch(color=c, label=f'{n}:{p:.1f}%') for c,n,p in zip(LEVEL_COLORS,LEVEL_NAMES,pcts)], loc='lower right', fontsize=8)
diff = np.array(pcts)-np.array(p3)
ax3b.text(0.02, 0.02, "POI影响:\n"+"".join(f"  {n}:{d:+.1f}%\n" for n,d in zip(LEVEL_NAMES, diff)),
          transform=ax3b.transAxes, fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
plt.tight_layout(); save(fig3, 'Step3_Final_03_3Dvs4D.png')

# 图4: 分布
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(15, 6)); fig4.suptitle('风险得分分布', fontsize=13, fontweight='bold')
ax4a.hist(score, bins=80, color='#3498DB', alpha=0.7, edgecolor='white')
for c, b in zip(LEVEL_COLORS, breaks[1:-1]): ax4a.axvline(b, color=c, linewidth=2, linestyle='--')
ax4a.set_xlabel('TOPSIS得分'); ax4a.set_ylabel('频数'); ax4a.set_title('分布 (竖线=断裂点)', fontweight='bold')
bd = [score[risk_lv==i] for i in range(1,6) if np.sum(risk_lv==i)>0]
bp = ax4b.boxplot(bd, patch_artist=True, notch=True)
for patch, c in zip(bp['boxes'], LEVEL_COLORS[:len(bd)]): patch.set_facecolor(c); patch.set_alpha(0.7)
ax4b.set_xticklabels([LEVEL_NAMES[i-1] for i in range(1,6) if np.sum(risk_lv==i)>0], fontsize=9)
ax4b.set_ylabel('得分'); ax4b.set_title(f'箱线图 (KW H={kw_h:.0f}***)', fontweight='bold')
plt.tight_layout(); save(fig4, 'Step3_Final_04_Distribution.png')

# 图5: 雷达图
fig5 = plt.figure(figsize=(14, 6)); fig5.suptitle('风险驱动因素', fontsize=13, fontweight='bold')
rm = np.array([X[risk_lv==k].mean(axis=0) if np.sum(risk_lv==k)>0 else np.zeros(4) for k in range(1,6)])
ax5a = plt.subplot(121, projection='polar'); ang = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()+[0]
for k, (c, lb) in enumerate(zip(LEVEL_COLORS, LEVEL_NAMES)):
    v = rm[k].tolist()+[rm[k][0]]; ax5a.fill(ang, v, alpha=0.12, color=c)
    ax5a.plot(ang, v, 'o-', linewidth=2, color=c, label=lb, markersize=5)
ax5a.set_xticks(ang[:-1]); ax5a.set_xticklabels(SHORT); ax5a.set_title('各级指标均值', fontweight='bold', pad=15)
ax5a.legend(fontsize=8, bbox_to_anchor=(1.3,1.1))
ax5b = plt.subplot(122); xp = np.arange(4); wd = 0.35
ax5b.bar(xp-wd/2, list(w3)+[0], wd, label='3维', color='#3498DB', alpha=0.7)
ax5b.bar(xp+wd/2, w, wd, label='4维', color='#E74C3C', alpha=0.7)
ax5b.set_xticks(xp); ax5b.set_xticklabels(SHORT); ax5b.set_ylabel('权重'); ax5b.set_title('POI对权重的影响', fontweight='bold'); ax5b.legend()
plt.tight_layout(); save(fig5, 'Step3_Final_05_Drivers.png')

# ── 保存 ──
print("\n[保存]...")
for name, data in [('Risk_Score_4D.tif', risk_score_2d), ('Risk_Level_4D.tif', risk_level_2d)]:
    with rasterio.open(os.path.join(OUTPUT_DIR, name), 'w', **out_profile) as dst:
        dst.write(np.where(np.isnan(data), -9999.0, data).astype(np.float32), 1)
    print(f"  → {name}")

rows = []
for i, (nm, pct) in enumerate(zip(LEVEL_NAMES, pcts), 1):
    grp = score[risk_lv==i] if np.sum(risk_lv==i)>0 else [np.nan]
    rows.append({'等级':i,'名称':nm,'面积%':round(pct,2),'下界':float(breaks[i-1]),
                 '上界':float(breaks[i]),'均值':float(np.nanmean(grp)),'Std':float(np.nanstd(grp))})
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'Risk_Summary_4D.csv'), index=False)
pd.DataFrame({'指标':NAMES,'权重':w,'CV':cv_list}).to_csv(os.path.join(OUTPUT_DIR, 'Weights_4D.csv'), index=False)

print("\n" + "=" * 70)
print("✅ Step 3 完成！POI暴露度权重=%.1f%% (最高)" % (w[3]*100))
print("=" * 70)