# ============================================================
# Week 5 — Reporting, Insights, and Presentation (Capstone)
# Consolidated pipeline reproducing every result and figure
# referenced in the final report.
# AI4I 2020 Predictive Maintenance Dataset
# Yuva Intern (NSDC) — Virtual Data Science with R Apprentice Intern
# ============================================================

suppressMessages({
  library(dplyr)
  library(ggplot2)
  library(randomForest)
  library(pROC)
})

set.seed(42)
dir.create("figures_final", showWarnings = FALSE)

# ---------- 1. Data Acquisition ----------
df <- read.csv("data/ai4i2020.csv", stringsAsFactors = FALSE)
names(df) <- c("UDI","ProductID","Type","AirTemp","ProcessTemp","RotSpeed","Torque","ToolWear",
               "MachineFailure","TWF","HDF","PWF","OSF","RNF")
df$Type <- factor(df$Type, levels = c("L","M","H"))
df$MachineFailure <- factor(ifelse(df$MachineFailure == 1, "Failure", "No_Failure"),
                             levels = c("No_Failure", "Failure"))

cat("=== DATA ACQUIRED ===\n")
cat("Rows:", nrow(df), " Cols:", ncol(df), "\n")
print(table(df$MachineFailure))

# ---------- 2. Key EDA Finding: Speed vs. Torque by Failure ----------
p_eda <- ggplot(df, aes(x = RotSpeed, y = Torque, color = MachineFailure)) +
  geom_point(alpha = 0.5, size = 1.3) +
  scale_color_manual(values = c("No_Failure" = "#A6A6A6", "Failure" = "#C00000")) +
  labs(title = "Rotational Speed vs. Torque, Colored by Failure Status",
       x = "Rotational Speed (rpm)", y = "Torque (Nm)", color = "Outcome") +
  theme_minimal(base_size = 13)
ggsave("figures_final/fig1_eda_speed_torque.png", p_eda, width = 7.5, height = 4.8, dpi = 150)

# ---------- 3. Cleaning + Feature Engineering (Week 3 logic) ----------
q <- quantile(df$Torque, c(0.25, 0.75)); iqr <- IQR(df$Torque)
lower <- q[1] - 1.5*iqr; upper <- q[2] + 1.5*iqr
df$Torque_capped <- pmin(pmax(df$Torque, lower), upper)

# Engineered feature: mechanical power, motivated directly by the EDA finding above
df$Power <- df$Torque_capped * df$RotSpeed * (2 * pi / 60)

scale_cols <- c("AirTemp","ProcessTemp","RotSpeed","Torque_capped","ToolWear","Power")
df[paste0(scale_cols, "_z")] <- scale(df[scale_cols])

onehot <- model.matrix(~ Type - 1, data = df)
colnames(onehot) <- gsub("Type", "Type_", colnames(onehot))
df <- cbind(df, onehot)

cat("\n=== FEATURES ENGINEERED: Power, z-scaled sensors, one-hot Type ===\n")

# ---------- 4. Model Training (Week 4 logic) ----------
model_vars <- c("AirTemp_z","ProcessTemp_z","RotSpeed_z","Torque_capped_z","ToolWear_z",
                 "Power_z","Type_L","Type_M","Type_H")
model_df <- df[, c(model_vars, "MachineFailure")]

idx_fail <- which(model_df$MachineFailure == "Failure")
idx_ok   <- which(model_df$MachineFailure == "No_Failure")
train_idx <- c(sample(idx_fail, round(0.7*length(idx_fail))),
               sample(idx_ok,   round(0.7*length(idx_ok))))
train_df <- model_df[train_idx, ]
test_df  <- model_df[-train_idx, ]

log_model <- glm(MachineFailure ~ ., data = train_df, family = binomial)
log_probs <- predict(log_model, newdata = test_df, type = "response")
log_roc <- roc(test_df$MachineFailure, log_probs, levels = c("No_Failure","Failure"), quiet = TRUE)

rf_model <- randomForest(MachineFailure ~ ., data = train_df, ntree = 300, mtry = 3, importance = TRUE)
rf_probs <- predict(rf_model, newdata = test_df, type = "prob")[, "Failure"]
rf_roc <- roc(test_df$MachineFailure, rf_probs, levels = c("No_Failure","Failure"), quiet = TRUE)

cat(sprintf("\n=== FINAL MODEL COMPARISON ===\nLogistic Regression AUC: %.4f\nRandom Forest AUC:       %.4f\n",
            auc(log_roc), auc(rf_roc)))

# ---------- 5. Final Report Figures ----------
png("figures_final/fig2_roc_curves.png", width = 1050, height = 850, res = 150)
plot(log_roc, col = "#2E74B5", lwd = 2.5, main = "ROC Curves: Logistic Regression vs. Random Forest")
plot(rf_roc, col = "#C00000", lwd = 2.5, add = TRUE)
legend("bottomright",
       legend = c(sprintf("Logistic Regression (AUC = %.3f)", auc(log_roc)),
                  sprintf("Random Forest (AUC = %.3f)", auc(rf_roc))),
       col = c("#2E74B5", "#C00000"), lwd = 2.5, bty = "n")
dev.off()

imp <- importance(rf_model)
imp_df <- data.frame(Variable = rownames(imp), Gini = imp[, "MeanDecreaseGini"])
imp_df <- imp_df[order(-imp_df$Gini), ]
imp_df$Variable <- factor(imp_df$Variable, levels = rev(imp_df$Variable))
p_imp <- ggplot(imp_df, aes(x = Variable, y = Gini)) +
  geom_col(fill = "#2E74B5", width = 0.65) + coord_flip() +
  labs(title = "Random Forest Variable Importance \u2014 Power Feature Ranks #1",
       x = NULL, y = "Mean Decrease in Gini") +
  theme_minimal(base_size = 13)
ggsave("figures_final/fig3_variable_importance.png", p_imp, width = 7, height = 4.5, dpi = 150)

cat("\n=== TOP PREDICTOR ===\n")
print(head(imp_df[order(-imp_df$Gini), ], 3))

cat("\nAll final report figures saved to figures_final/\n")
cat("\nThis script reproduces every figure and metric cited in the Week 5 capstone report.\n")
