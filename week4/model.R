# ============================================================
# Week 4 — Model Building and Predictive Analysis
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

# ---------- 0. Load processed data from Week 3 ----------
df <- read.csv("data/ai4i2020_processed.csv", stringsAsFactors = FALSE)
df$MachineFailure <- factor(ifelse(df$MachineFailure == 1, "Failure", "No_Failure"),
                             levels = c("No_Failure", "Failure"))
df$Type <- factor(df$Type, levels = c("L","M","H"))

cat("=== PROBLEM: PREDICT MachineFailure ===\n")
cat("Rows:", nrow(df), " | Columns:", ncol(df), "\n")
print(table(df$MachineFailure))
cat(sprintf("Failure rate: %.2f%%\n", 100*mean(df$MachineFailure == "Failure")))

# ---------- 1. Feature selection for modelling ----------
model_vars <- c("AirTemp_z","ProcessTemp_z","RotSpeed_z","Torque_capped_z","ToolWear_z",
                 "Power_z","Type_L","Type_M","Type_H")
model_df <- df[, c(model_vars, "MachineFailure")]

cat("\n=== MODELLING FEATURES ===\n")
print(names(model_df))
cat("\n=== MISSING VALUES CHECK ===\n")
print(colSums(is.na(model_df)))

# ---------- 2. Train / Test split (stratified 70/30) ----------
idx_fail <- which(model_df$MachineFailure == "Failure")
idx_ok   <- which(model_df$MachineFailure == "No_Failure")

train_fail <- sample(idx_fail, size = round(0.7 * length(idx_fail)))
train_ok   <- sample(idx_ok,   size = round(0.7 * length(idx_ok)))
train_idx  <- c(train_fail, train_ok)

train_df <- model_df[train_idx, ]
test_df  <- model_df[-train_idx, ]

cat("\n=== TRAIN / TEST SPLIT (stratified 70/30) ===\n")
cat("Train:", nrow(train_df), " (failures:", sum(train_df$MachineFailure=="Failure"), ")\n")
cat("Test: ", nrow(test_df),  " (failures:", sum(test_df$MachineFailure=="Failure"), ")\n")

# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================
cat("\n\n################ MODEL 1: LOGISTIC REGRESSION ################\n")
log_model <- glm(MachineFailure ~ ., data = train_df, family = binomial)
cat("\n=== MODEL SUMMARY (coefficients) ===\n")
print(summary(log_model)$coefficients)

log_probs <- predict(log_model, newdata = test_df, type = "response")
log_pred  <- factor(ifelse(log_probs > 0.5, "Failure", "No_Failure"), levels = c("No_Failure","Failure"))

log_cm <- table(Predicted = log_pred, Actual = test_df$MachineFailure)
cat("\n=== CONFUSION MATRIX (threshold = 0.5) ===\n")
print(log_cm)

log_acc  <- sum(diag(log_cm)) / sum(log_cm)
log_prec <- log_cm["Failure","Failure"] / sum(log_cm["Failure",])
log_rec  <- log_cm["Failure","Failure"] / sum(log_cm[,"Failure"])
log_f1   <- 2 * log_prec * log_rec / (log_prec + log_rec)
log_roc  <- roc(test_df$MachineFailure, log_probs, levels = c("No_Failure","Failure"), quiet = TRUE)
log_auc  <- as.numeric(auc(log_roc))

cat(sprintf("\nAccuracy:  %.4f\nPrecision: %.4f\nRecall:    %.4f\nF1 Score:  %.4f\nAUC:       %.4f\n",
            log_acc, log_prec, log_rec, log_f1, log_auc))

# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================
cat("\n\n################ MODEL 2: RANDOM FOREST ################\n")
rf_model <- randomForest(MachineFailure ~ ., data = train_df, ntree = 300,
                          mtry = 3, importance = TRUE)
print(rf_model)

rf_probs <- predict(rf_model, newdata = test_df, type = "prob")[, "Failure"]
rf_pred  <- predict(rf_model, newdata = test_df, type = "class")

rf_cm <- table(Predicted = rf_pred, Actual = test_df$MachineFailure)
cat("\n=== CONFUSION MATRIX (threshold = 0.5) ===\n")
print(rf_cm)

rf_acc  <- sum(diag(rf_cm)) / sum(rf_cm)
rf_prec <- rf_cm["Failure","Failure"] / sum(rf_cm["Failure",])
rf_rec  <- rf_cm["Failure","Failure"] / sum(rf_cm[,"Failure"])
rf_f1   <- 2 * rf_prec * rf_rec / (rf_prec + rf_rec)
rf_roc  <- roc(test_df$MachineFailure, rf_probs, levels = c("No_Failure","Failure"), quiet = TRUE)
rf_auc  <- as.numeric(auc(rf_roc))

cat(sprintf("\nAccuracy:  %.4f\nPrecision: %.4f\nRecall:    %.4f\nF1 Score:  %.4f\nAUC:       %.4f\n",
            rf_acc, rf_prec, rf_rec, rf_f1, rf_auc))

cat("\n=== RANDOM FOREST VARIABLE IMPORTANCE ===\n")
imp <- importance(rf_model)
print(imp[order(-imp[,"MeanDecreaseGini"]), ])

# ============================================================
# COMPARISON TABLE
# ============================================================
cat("\n\n################ MODEL COMPARISON ################\n")
comparison <- data.frame(
  Model = c("Logistic Regression","Random Forest"),
  Accuracy = c(log_acc, rf_acc),
  Precision = c(log_prec, rf_prec),
  Recall = c(log_rec, rf_rec),
  F1 = c(log_f1, rf_f1),
  AUC = c(log_auc, rf_auc)
)
print(comparison, digits = 4)
write.csv(comparison, "model_comparison.csv", row.names = FALSE)

# ============================================================
# FIGURES
# ============================================================
dir.create("figures_w4", showWarnings = FALSE)

# ROC curves (both models on one plot)
png("figures_w4/fig1_roc_curves.png", width = 1050, height = 850, res = 150)
plot(log_roc, col = "#2E74B5", lwd = 2.5, main = "ROC Curves: Logistic Regression vs. Random Forest")
plot(rf_roc, col = "#C00000", lwd = 2.5, add = TRUE)
legend("bottomright",
       legend = c(sprintf("Logistic Regression (AUC = %.3f)", log_auc),
                  sprintf("Random Forest (AUC = %.3f)", rf_auc)),
       col = c("#2E74B5", "#C00000"), lwd = 2.5, bty = "n")
dev.off()

# Variable importance plot
imp_df <- data.frame(Variable = rownames(imp), Gini = imp[, "MeanDecreaseGini"])
imp_df <- imp_df[order(-imp_df$Gini), ]
imp_df$Variable <- factor(imp_df$Variable, levels = rev(imp_df$Variable))
p2 <- ggplot(imp_df, aes(x = Variable, y = Gini)) +
  geom_col(fill = "#2E74B5", width = 0.65) +
  coord_flip() +
  labs(title = "Random Forest Variable Importance", x = NULL, y = "Mean Decrease in Gini") +
  theme_minimal(base_size = 13)
ggsave("figures_w4/fig2_variable_importance.png", p2, width = 7, height = 4.5, dpi = 150)

# Confusion matrix heatmaps
cm_to_df <- function(cm, model_name) {
  d <- as.data.frame(cm)
  d$Model <- model_name
  d
}
cm_all <- rbind(cm_to_df(log_cm, "Logistic Regression"), cm_to_df(rf_cm, "Random Forest"))
p3 <- ggplot(cm_all, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, fontface = "bold") +
  scale_fill_gradient(low = "#DCE6F1", high = "#1F3864") +
  facet_wrap(~Model) +
  labs(title = "Confusion Matrices (Test Set, Threshold = 0.5)") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")
ggsave("figures_w4/fig3_confusion_matrices.png", p3, width = 8, height = 4.3, dpi = 150)

# Precision-Recall style: predicted probability distribution by actual class (RF)
prob_df <- data.frame(Prob = rf_probs, Actual = test_df$MachineFailure)
p4 <- ggplot(prob_df, aes(x = Prob, fill = Actual)) +
  geom_histogram(bins = 30, color = "white") +
  scale_fill_manual(values = c("No_Failure" = "#A6A6A6", "Failure" = "#C00000")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "black") +
  facet_wrap(~Actual, scales = "free_y", ncol = 1) +
  labs(title = "RF Predicted Failure Probability by Actual Outcome",
       x = "Predicted P(Failure)", y = "Count") +
  theme_minimal(base_size = 13) + theme(legend.position = "none")
ggsave("figures_w4/fig4_predicted_prob_distribution.png", p4, width = 7, height = 5.2, dpi = 150)

cat("\nAll figures saved to figures_w4/\n")
cat("\n=== SESSION INFO ===\n")
print(sessionInfo())
