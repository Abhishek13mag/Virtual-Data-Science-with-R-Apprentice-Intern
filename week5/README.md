# Week 5 — Reporting, Insights, and Presentation (Capstone)

Yuva Intern (NSDC) — Virtual Data Science with R Apprentice Intern, Week 5 of 5.

## Overview
Final capstone report synthesizing the full predictive-maintenance project on the
AI4I 2020 dataset: EDA → data wrangling → feature engineering → model building →
insights and recommendations.

## Contents
- `capstone.R` — consolidated script reproducing every figure and metric cited
  in the final report (data load → clean → engineer Power feature → train
  Logistic Regression + Random Forest → export final figures)
- `data/ai4i2020.csv` — source dataset
- `figures_final/` — the 3 key figures referenced in the report (EDA scatter,
  ROC curves, variable importance)
- `Week5_Final_Project_Report.docx` — the full written capstone report

## How to Run
```r
install.packages(c("dplyr", "ggplot2", "randomForest", "pROC"))
source("capstone.R")
```

## Key Result
Random Forest (AUC ≈ 0.96) outperforms Logistic Regression (AUC ≈ 0.92), and
the engineered `Power` feature — created directly from an EDA finding — ranks
as the single most important predictor, validating the project's EDA-driven
feature engineering approach end to end.

## Author
Abhishek
