# Week 3 — Data Wrangling and Preprocessing in R

Yuva Intern (NSDC) — Virtual Data Science with R Apprentice Intern, Week 3 of 5.

## Overview
Reproducible data wrangling pipeline on the **AI4I 2020 Predictive Maintenance
Dataset**, continuing from Weeks 1–2. Since the source data is clean by default,
a documented set of realistic issues (missing values, inconsistent labels, a
type-corruption) is injected under `set.seed(42)` to demonstrate cleaning
techniques on real data, fully reproducibly.

## Contents
- `wrangle.R` — full pipeline: issue injection → quality assessment → cleaning → transformation → export
- `data/ai4i2020.csv` — original source dataset
- `data/ai4i2020_processed.csv` — final cleaned, encoded, and scaled output (27 columns)
- `figures_w3/` — missingness chart, before/after distribution, scaling effect
- `Week3_Data_Wrangling_Report.docx` — full written report with code, output, and interpretation

## How to Run
```r
install.packages(c("dplyr", "tidyr", "ggplot2"))
source("wrangle.R")
```

## Pipeline Steps
1. **Quality assessment** — missing values, duplicates, label consistency, IQR outliers
2. **Cleaning** — type correction (regex + coercion), label standardization (trim/case), imputation (mean for AirTemp, median for Torque/ToolWear), outlier winsorizing
3. **Transformation** — feature extraction (Power, ToolWearRisk), ordinal + one-hot encoding of Type, z-score scaling of all continuous sensors
4. **Export** — processed dataset written to `data/ai4i2020_processed.csv`

## Author
Abhishek
