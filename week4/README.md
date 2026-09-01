# Week 4 — Model Building and Predictive Analysis in R

Yuva Intern (NSDC) — Virtual Data Science with R Apprentice Intern, Week 4 of 5.

## Overview
Binary classification of machine failure on the AI4I 2020 Predictive Maintenance
Dataset, using the cleaned/feature-engineered output from Week 3. Two models —
Logistic Regression and Random Forest — are trained, validated, and compared.

## Contents
- `model.R` — full pipeline: load processed data → stratified split → train both models → evaluate → export figures
- `data/ai4i2020_processed.csv` — Week 3 output (27 columns, cleaned/encoded/scaled)
- `model_comparison.csv` — final metrics table (accuracy, precision, recall, F1, AUC)
- `figures_w4/` — ROC curves, variable importance, confusion matrices, probability distribution
- `Week4_Model_Building_Report.docx` — full written report with code, output, and interpretation

## How to Run
```r
install.packages(c("dplyr", "ggplot2", "randomForest", "pROC"))
source("model.R")
```

## Results Summary
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.971 | 0.703 | 0.255 | 0.374 | 0.928 |
| Random Forest | 0.986 | 0.883 | 0.667 | 0.760 | 0.961 |

Random Forest outperforms on every imbalance-sensitive metric. The engineered
`Power` feature (from Week 3) ranks as the single most important predictor.

## Author
Abhishek
