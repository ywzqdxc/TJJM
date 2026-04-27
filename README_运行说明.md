# 北京市内涝风险区划 — 完整运行说明

## 目录结构（运行前先建好）

```
项目根目录/
├── data_input/
│   ├── 北京市_DEM_30m分辨率_NASA数据.tif
│   ├── CHM_PRE_V2_daily_2018.nc        ← 2018年降雨NC
│   ├── CHM_PRE_V2_daily_2019.nc
│   ├── ... (2018-2024，共7个)
│   ├── CHM_PRE_V2_daily_2024.nc
│   ├── fac/
│   │   ├── K_SCH_Aligned_30m.tif
│   │   ├── PSI_Aligned_30m.tif
│   │   └── THSCH_Aligned_30m.tif
│   └── Processed_Decadal_SM_UTM50N/
│       ├── 2018/
│       │   ├── SM_2018_06_1_Mean_30m.tif
│       │   └── ... (每年11个旬均值文件)
│       └── ... (2019-2024)
├── flood_labels_clean.csv              ← 已生成，直接放这里
├── step1_extract_rainfall.py
├── step2_extract_dem_features.py
├── step3_extract_soil_moisture.py
├── step4_build_dataset.py
├── step5_statistical_model.py
├── step6_risk_mapping.py
├── data_processed/                     ← 自动创建
└── results/                            ← 自动创建
```

## 运行顺序（按序执行，每步依赖上一步输出）

```bash
# 安装依赖（一次性）
pip install rasterio pysheds xarray netCDF4 geopandas shapely scikit-learn matplotlib scipy

# 按顺序运行
python step1_extract_rainfall.py      # ~5-10分钟
python step2_extract_dem_features.py  # ~10-20分钟（HAND计算较慢）
python step3_extract_soil_moisture.py # ~5-10分钟
python step4_build_dataset.py         # ~1分钟（合并构造样本）
python step5_statistical_model.py     # ~2-3分钟（建模+检验）
python step6_risk_mapping.py          # ~5分钟（出图）
```

## 各脚本产出文件

| 步骤 | 产出文件 | 说明 |
|------|----------|------|
| Step1 | data_processed/rainfall_daily_2018_2024.csv | 逐日降雨量 |
| Step2 | data_processed/dem_soil_features.csv | 地形+土壤特征 |
| Step3 | data_processed/soil_moisture_daily_2018_2024.csv | 前期土壤水分 |
| Step4 | data_processed/All_Years_Dataset.csv | **建模数据集** |
| Step5 | results/model_comparison.csv | 三模型对比 |
| Step5 | results/logistic_coefficients.csv | 回归系数+OR+CI |
| Step5 | results/feature_importance.csv | 特征重要性 |
| Step5 | results/robustness_check.csv | 稳健性检验 |
| Step5 | results/model_results.png | ROC+重要性图 |
| Step6 | results/risk_map.png | 五级风险区划图 |
| Step6 | results/district_topsis_ranking.csv | 区县TOPSIS排名 |
| Step6 | results/topsis_ranking.png | 区县排名图 |

## 三人分工对应关系

| 队员 | 负责脚本 | 产出 |
|------|----------|------|
| **A — 数据特征工程** | step1 + step2 + step3 + step4 | All_Years_Dataset.csv |
| **B — 统计建模检验** | step5 | 所有results/表格 |
| **C — 区划与论文** | step6 + 论文撰写 | 风险图+论文 |

## 注意事项

1. **Step2 HAND计算**：全市36M像元计算HAND约需10-20分钟，内存需>=8GB
   如内存不足，可在脚本中调整 `acc > 500` 阈值为 `acc > 2000`（减少河道像元）

2. **Step1 NC文件路径**：确认NC文件名格式为 `CHM_PRE_V2_daily_YYYY.nc`
   如格式不同，修改 `step1` 中 `nc_file = f'CHM_PRE_V2_daily_{year}.nc'` 这行

3. **Step3 SM文件路径**：确认旬均文件命名为 `SM_YYYY_MM_D_Mean_30m.tif`
   D=1(上旬)/2(中旬)/3(下旬)，如命名不同修改 `step3` 中的文件名拼接逻辑

4. **论文中统计检验汇报顺序**（大赛评分关键）：
   - 数据检验：VIF → 相关分析
   - 模型检验：Hosmer-Lemeshow → P值/OR值/95%CI
   - 结果检验：AUC-ROC → KS统计量 → 稳健性分析
