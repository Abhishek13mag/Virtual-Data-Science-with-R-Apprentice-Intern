# ============================================================
# Week 2 — Exploratory Data Analysis & Visualization
# AI4I 2020 Predictive Maintenance Dataset
# Yuva Intern (NSDC) — Virtual Data Science with R Apprentice Intern
# ============================================================

suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(reshape2)
  library(corrplot)
})

# ---------- 1. Load & Clean ----------
df <- read.csv("data/ai4i2020.csv", stringsAsFactors = FALSE)
names(df) <- c("UDI","ProductID","Type","AirTemp","ProcessTemp","RotSpeed","Torque","ToolWear",
               "MachineFailure","TWF","HDF","PWF","OSF","RNF")

cat("=== STRUCTURE ===\n")
str(df)

cat("\n=== MISSING VALUES PER COLUMN ===\n")
print(colSums(is.na(df)))

cat("\n=== DUPLICATE ROWS ===\n")
print(sum(duplicated(df)))

df$Type <- factor(df$Type, levels = c("L","M","H"))
df$MachineFailure <- factor(df$MachineFailure, labels = c("No Failure","Failure"))

# ---------- 2. Summary Statistics ----------
num_cols <- c("AirTemp","ProcessTemp","RotSpeed","Torque","ToolWear")
cat("\n=== SUMMARY STATISTICS ===\n")
print(summary(df[, num_cols]))

cat("\n=== CLASS BALANCE: Machine Failure ===\n")
tab <- table(df$MachineFailure)
print(tab)
cat(sprintf("Failure rate: %.2f%%\n", 100 * tab["Failure"] / sum(tab)))

cat("\n=== FAILURE RATE BY PRODUCT TYPE ===\n")
print(round(prop.table(table(df$Type, df$MachineFailure), margin = 1) * 100, 2))

cat("\n=== OUTLIER CHECK (IQR method, Torque) ===\n")
q <- quantile(df$Torque, c(0.25, 0.75))
iqr <- IQR(df$Torque)
outliers <- df$Torque[df$Torque < (q[1] - 1.5*iqr) | df$Torque > (q[2] + 1.5*iqr)]
cat(sprintf("Number of outlier Torque readings: %d\n", length(outliers)))

cat("\n=== CORRELATION MATRIX ===\n")
cor_mat <- cor(df[, num_cols])
print(round(cor_mat, 3))

# ---------- 3. Visualizations ----------
dir.create("figures", showWarnings = FALSE)

# Fig 1: Histogram
p1 <- ggplot(df, aes(x = RotSpeed)) +
  geom_histogram(bins = 40, fill = "#2E74B5", color = "white") +
  labs(title = "Distribution of Rotational Speed", x = "Rotational Speed (rpm)", y = "Count") +
  theme_minimal(base_size = 13)
ggsave("figures/fig1_histogram_rotspeed.png", p1, width = 7, height = 4.5, dpi = 150)

# Fig 2: Boxplot
p2 <- ggplot(df, aes(x = Type, y = Torque, fill = Type)) +
  geom_boxplot(alpha = 0.85, outlier.color = "#C00000") +
  scale_fill_manual(values = c("L" = "#9DC3E6", "M" = "#2E74B5", "H" = "#1F3864")) +
  labs(title = "Torque Distribution by Product Quality Type", x = "Product Type", y = "Torque (Nm)") +
  theme_minimal(base_size = 13) + theme(legend.position = "none")
ggsave("figures/fig2_boxplot_torque_type.png", p2, width = 7, height = 4.5, dpi = 150)

# Fig 3: Scatter plot
p3 <- ggplot(df, aes(x = RotSpeed, y = Torque, color = MachineFailure)) +
  geom_point(alpha = 0.5, size = 1.3) +
  scale_color_manual(values = c("No Failure" = "#A6A6A6", "Failure" = "#C00000")) +
  labs(title = "Rotational Speed vs. Torque, Colored by Failure Status",
       x = "Rotational Speed (rpm)", y = "Torque (Nm)", color = "Outcome") +
  theme_minimal(base_size = 13)
ggsave("figures/fig3_scatter_speed_torque.png", p3, width = 7.5, height = 4.8, dpi = 150)

# Fig 4: Correlation heatmap
png("figures/fig4_correlation_heatmap.png", width = 1100, height = 950, res = 150)
corrplot(cor_mat, method = "color", type = "upper", addCoef.col = "black",
         tl.col = "black", tl.srt = 45, number.cex = 0.8,
         col = colorRampPalette(c("#C00000","#FFFFFF","#2E74B5"))(200),
         title = "Correlation Matrix: Sensor Variables", mar = c(0,0,2,0))
dev.off()

# Fig 5: Bar chart — failure rate by type
fail_by_type <- df %>%
  group_by(Type) %>%
  summarise(FailureRate = mean(MachineFailure == "Failure") * 100)
p5 <- ggplot(fail_by_type, aes(x = Type, y = FailureRate, fill = Type)) +
  geom_col(width = 0.6) +
  scale_fill_manual(values = c("L" = "#9DC3E6", "M" = "#2E74B5", "H" = "#1F3864")) +
  labs(title = "Machine Failure Rate by Product Quality Type", x = "Product Type", y = "Failure Rate (%)") +
  theme_minimal(base_size = 13) + theme(legend.position = "none") +
  geom_text(aes(label = sprintf("%.2f%%", FailureRate)), vjust = -0.5, size = 4)
ggsave("figures/fig5_bar_failure_by_type.png", p5, width = 6.5, height = 4.5, dpi = 150)

cat("\nAll figures saved to figures/\n")
