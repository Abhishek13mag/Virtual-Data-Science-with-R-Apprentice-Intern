# ============================================================
# Week 3 — Data Wrangling and Preprocessing
# AI4I 2020 Predictive Maintenance Dataset
# Yuva Intern (NSDC) — Virtual Data Science with R Apprentice Intern
# ============================================================

suppressMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

set.seed(42)  # reproducibility for the synthetic-issue injection step

# ---------- 0. Load ----------
raw <- read.csv("data/ai4i2020.csv", stringsAsFactors = FALSE)
names(raw) <- c("UDI","ProductID","Type","AirTemp","ProcessTemp","RotSpeed","Torque","ToolWear",
                "MachineFailure","TWF","HDF","PWF","OSF","RNF")

cat("=== BEFORE: STRUCTURE ===\n")
str(raw)

cat("\n=== BEFORE: FIRST 5 ROWS ===\n")
print(head(raw, 5))

cat("\n=== BEFORE: MISSING VALUES ===\n")
print(colSums(is.na(raw)))

cat("\n=== BEFORE: DUPLICATE ROWS ===\n")
print(sum(duplicated(raw)))

cat("\n=== BEFORE: TYPE FIELD VALUE CONSISTENCY ===\n")
print(table(raw$Type))

# ------------------------------------------------------------
# The source file is a clean, competition-grade release with no
# missing values, duplicates, or label inconsistencies. To
# demonstrate the full set of wrangling techniques required for
# this task (missing-value handling, type correction, outlier
# treatment, encoding, scaling) on real data, we deliberately
# inject a controlled, documented set of typical real-world data
# quality issues under a fixed random seed (42) for reproducibility.
# This step is clearly separated from genuine cleaning below.
# ------------------------------------------------------------

df <- raw

# Inject ~3% MCAR missingness into two continuous sensor columns
n <- nrow(df)
miss_idx_air   <- sample(seq_len(n), size = round(0.03 * n))
miss_idx_torque <- sample(seq_len(n), size = round(0.03 * n))
df$AirTemp[miss_idx_air]   <- NA
df$Torque[miss_idx_torque] <- NA

# Inject inconsistent categorical labels (case/whitespace issues)
inconsist_idx <- sample(seq_len(n), size = round(0.02 * n))
df$Type[inconsist_idx] <- ifelse(df$Type[inconsist_idx] == "L", "l ",
                            ifelse(df$Type[inconsist_idx] == "M", " m", "h"))

# Introduce a data-type issue: ToolWear read in as character with stray text
type_issue_idx <- sample(seq_len(n), size = round(0.01 * n))
df$ToolWear <- as.character(df$ToolWear)
df$ToolWear[type_issue_idx] <- paste0(df$ToolWear[type_issue_idx], " min")

cat("\n\n################ DATA QUALITY ASSESSMENT (POST-INJECTION) ################\n")
cat("\n=== MISSING VALUES AFTER INJECTION ===\n")
print(colSums(is.na(df)))

cat("\n=== INCONSISTENT CATEGORY LABELS (Type) ===\n")
print(table(df$Type))

cat("\n=== TOOLWEAR SAMPLE SHOWING TYPE ISSUE ===\n")
print(head(df$ToolWear[type_issue_idx], 5))

cat("\n=== OUTLIER CHECK: Torque (IQR method, on non-missing values) ===\n")
q <- quantile(df$Torque, c(0.25, 0.75), na.rm = TRUE)
iqr <- IQR(df$Torque, na.rm = TRUE)
lower <- q[1] - 1.5 * iqr
upper <- q[2] + 1.5 * iqr
n_outliers <- sum(df$Torque < lower | df$Torque > upper, na.rm = TRUE)
cat(sprintf("Bounds: [%.2f, %.2f] | Outlier count: %d\n", lower, upper, n_outliers))

# ============================================================
# DATA CLEANING
# ============================================================
cat("\n\n################ DATA CLEANING ################\n")

# --- Step 1: Correct data types ---
df$ToolWear <- as.numeric(gsub("[^0-9.]", "", df$ToolWear))
cat("\nStep 1 — ToolWear coerced back to numeric. New NAs introduced by coercion:",
    sum(is.na(df$ToolWear)) - sum(is.na(raw$ToolWear)), "\n")

# --- Step 2: Standardize categorical labels ---
df$Type <- trimws(toupper(df$Type))
df$Type <- factor(df$Type, levels = c("L","M","H"))
cat("\nStep 2 — Type standardized (trimmed + uppercased). Value counts:\n")
print(table(df$Type, useNA = "ifany"))

# --- Step 3: Handle missing values ---
# AirTemp: near-normal, low skew -> mean imputation is appropriate
# Torque: mildly skewed with known outliers -> median imputation is more robust
airtemp_before <- df$AirTemp
torque_before  <- df$Torque

df$AirTemp[is.na(df$AirTemp)] <- mean(df$AirTemp, na.rm = TRUE)
df$Torque[is.na(df$Torque)]   <- median(df$Torque, na.rm = TRUE)
df$ToolWear[is.na(df$ToolWear)] <- median(df$ToolWear, na.rm = TRUE)

cat("\nStep 3 — Missing values imputed (AirTemp: mean, Torque & ToolWear: median).\n")
cat("Remaining missing values:\n")
print(colSums(is.na(df)))

# --- Step 4: Outlier treatment (winsorize, don't delete) ---
# Week 2's EDA showed Torque outliers correlate with real failure events,
# so we cap (winsorize) rather than remove them, preserving the signal
# while limiting leverage on later scaling/model steps.
df$Torque_capped <- pmin(pmax(df$Torque, lower), upper)
n_capped <- sum(df$Torque != df$Torque_capped)
cat(sprintf("\nStep 4 — Torque winsorized at [%.2f, %.2f]. Values capped: %d\n", lower, upper, n_capped))

cat("\n=== AFTER CLEANING: FIRST 5 ROWS (key columns) ===\n")
print(head(df[, c("UDI","Type","AirTemp","Torque","Torque_capped","ToolWear")], 5))

# ============================================================
# DATA TRANSFORMATION
# ============================================================
cat("\n\n################ DATA TRANSFORMATION ################\n")

# --- Feature extraction: mechanical power proxy (from Week 2 insight) ---
df$Power <- df$Torque_capped * df$RotSpeed * (2 * pi / 60)  # approx watts
cat("\nExtracted feature: Power (Torque x Angular Velocity), Week 2 finding operationalized.\n")
print(summary(df$Power))

# --- Feature extraction: Tool wear risk bucket ---
df$ToolWearRisk <- cut(df$ToolWear, breaks = c(-1, 50, 150, Inf),
                        labels = c("Low","Medium","High"))
cat("\nExtracted feature: ToolWearRisk bucket. Distribution:\n")
print(table(df$ToolWearRisk))

# --- Encoding: ordinal label encoding for Type (L < M < H is meaningful order) ---
df$Type_ordinal <- as.integer(factor(df$Type, levels = c("L","M","H"), ordered = TRUE))

# --- Encoding: one-hot encoding for Type (for models that assume no ordering) ---
onehot <- model.matrix(~ Type - 1, data = df)
colnames(onehot) <- gsub("Type", "Type_", colnames(onehot))
df <- cbind(df, onehot)
cat("\nEncoding — Type_ordinal (label) and Type_L/Type_M/Type_H (one-hot) created.\n")
print(head(df[, c("Type","Type_ordinal","Type_L","Type_M","Type_H")], 5))

# --- Scaling: z-score standardization of continuous sensors ---
scale_cols <- c("AirTemp","ProcessTemp","RotSpeed","Torque_capped","ToolWear","Power")
df_scaled <- df
df_scaled[paste0(scale_cols, "_z")] <- scale(df[scale_cols])

cat("\nScaling — z-score standardized columns added (suffix _z). Check mean~0, sd~1:\n")
print(round(sapply(df_scaled[paste0(scale_cols, "_z")], function(x) c(mean = mean(x), sd = sd(x))), 4))

# ============================================================
# BEFORE / AFTER SNAPSHOT
# ============================================================
cat("\n\n################ BEFORE / AFTER SUMMARY ################\n")
cat("\nBEFORE (raw):\n")
print(head(raw[, c("UDI","Type","AirTemp","Torque","ToolWear")], 5))
cat("\nAFTER (cleaned + transformed):\n")
print(head(df_scaled[, c("UDI","Type","AirTemp","Torque_capped","ToolWear","Power","Type_ordinal","AirTemp_z")], 5))

cat(sprintf("\nDimensions — raw: %d cols | processed: %d cols\n", ncol(raw), ncol(df_scaled)))

# ---------- Figures ----------
dir.create("figures_w3", showWarnings = FALSE)

# Missingness pattern (before cleaning)
miss_df <- data.frame(
  Column = c("AirTemp","Torque","ToolWear"),
  Missing = c(sum(is.na(airtemp_before)), sum(is.na(torque_before)), sum(type_issue_idx %in% type_issue_idx))
)
p1 <- ggplot(miss_df, aes(x = Column, y = Missing, fill = Column)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  scale_fill_manual(values = c("#C00000","#2E74B5","#BF8F00")) +
  geom_text(aes(label = Missing), vjust = -0.4, size = 4) +
  labs(title = "Injected Data Quality Issues by Column", x = NULL, y = "Count") +
  theme_minimal(base_size = 13)
ggsave("figures_w3/fig1_missingness.png", p1, width = 7.5, height = 4.3, dpi = 150)

# Before/after distribution of Torque (imputation + winsorizing effect)
comp_df <- data.frame(
  Torque = c(torque_before[!is.na(torque_before)], df$Torque_capped),
  Stage = rep(c("Before (raw, NA removed)", "After (imputed + winsorized)"),
              c(sum(!is.na(torque_before)), nrow(df)))
)
p2 <- ggplot(comp_df, aes(x = Torque, fill = Stage)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#A6A6A6", "#2E74B5")) +
  labs(title = "Torque Distribution: Before vs. After Cleaning", x = "Torque (Nm)", y = "Density") +
  theme_minimal(base_size = 13) + theme(legend.position = "bottom", legend.title = element_blank())
ggsave("figures_w3/fig2_before_after_torque.png", p2, width = 7, height = 4.5, dpi = 150)

# Scaled vs unscaled comparison (boxplots on same axis, faceted)
scale_check <- data.frame(
  Value = c(df$AirTemp, df$Torque_capped, df_scaled$AirTemp_z, df_scaled$Torque_capped_z),
  Variable = rep(c("AirTemp (raw)","Torque (raw)","AirTemp (z-scaled)","Torque (z-scaled)"), each = n)
)
p3 <- ggplot(scale_check, aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = c("#9DC3E6","#2E74B5","#F4B183","#C00000")) +
  labs(title = "Effect of Z-Score Standardization on Scale", x = NULL, y = "Value") +
  theme_minimal(base_size = 12) + theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave("figures_w3/fig3_scaling_effect.png", p3, width = 7.5, height = 4.6, dpi = 150)

# Encoding illustration
enc_df <- head(df[, c("Type","Type_ordinal","Type_L","Type_M","Type_H")], 6)
cat("\nAll figures saved to figures_w3/\n")

# ---------- Save processed dataset ----------
write.csv(df_scaled, "data/ai4i2020_processed.csv", row.names = FALSE)
cat("\nProcessed dataset written to data/ai4i2020_processed.csv\n")

cat("\n=== SESSION INFO ===\n")
print(sessionInfo())
