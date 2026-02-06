
# Visualization Script (R)
# This script creates all visualizations for the house classification project
# Data and models are prepared by Python scripts

cat("============================================================\n")
cat("R VISUALIZATION SCRIPT\n")
cat("============================================================\n\n")

# Load required libraries
suppressPackageStartupMessages({
  library(ggplot2)
  library(reshape2)
  library(gridExtra)
})

# Create visualizations directory
if (!dir.exists("visualizations")) {
  dir.create("visualizations")
  cat("Created visualizations/directory\n")
}

# Load data
cat("Loading data from data/directory...\n")
house_data <- read.csv("data/house_data.csv")
test_predictions <- read.csv("data/test_predictions.csv")
coefficients <- read.csv("data/model_coefficients.csv", row.names=1)

cat(sprintf("Loaded %d house records\n", nrow(house_data)))
cat(sprintf("Loaded %d test predictions\n", nrow(test_predictions)))
cat(sprintf("Loaded coefficients for %d zip codes\n", nrow(coefficients)))

# Convert zip_code to factor
house_data$zip_code <- as.factor(house_data$zip_code)
test_predictions$actual <- as.factor(test_predictions$actual)
test_predictions$predicted <- as.factor(test_predictions$predicted)

cat("\n============================================================\n")
cat("CREATING VISUALIZATIONS\n")
cat("============================================================\n\n")

# ============================================================
# 1. EXPLORATORY DATA ANALYSIS PLOTS
# ============================================================

cat("Creating EDA plots...\n")

# Price distribution by zip code
p1 <- ggplot(house_data, aes(x=zip_code, y=price, fill=zip_code)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(labels = scales::comma) +
  theme_minimal() +
  labs(title="Price Distribution by Zip Code", 
       x="Zip Code", 
       y="Price ($)") +
  theme(legend.position="none",
        plot.title = element_text(face="bold", hjust=0.5))

# Square footage distribution
p2 <- ggplot(house_data, aes(x=sqft, fill=zip_code)) +
  geom_density(alpha=0.5) +
  theme_minimal() +
  labs(title="Square Footage Distribution", 
       x="Square Feet", 
       y="Density",
       fill="Zip Code") +
  theme(plot.title = element_text(face="bold", hjust=0.5))

# Price vs Square Footage
p3 <- ggplot(house_data, aes(x=sqft, y=price, color=zip_code)) +
  geom_point(alpha=0.6, size=2) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed") +
  scale_y_continuous(labels = scales::comma) +
  theme_minimal() +
  labs(title="Price vs Square Footage", 
       x="Square Feet", 
       y="Price ($)",
       color="Zip Code") +
  theme(plot.title = element_text(face="bold", hjust=0.5))

# Bedroom distribution
p4 <- ggplot(house_data, aes(x=as.factor(beds), fill=zip_code)) +
  geom_bar(position="dodge", alpha=0.7) +
  theme_minimal() +
  labs(title="Bedroom Distribution by Zip Code", 
       x="Number of Bedrooms", 
       y="Count",
       fill="Zip Code") +
  theme(plot.title = element_text(face="bold", hjust=0.5))

# Bathroom distribution
p5 <- ggplot(house_data, aes(x=baths, fill=zip_code)) +
  geom_histogram(position="dodge", bins=15, alpha=0.7) +
  theme_minimal() +
  labs(title="Bathroom Distribution by Zip Code", 
       x="Number of Bathrooms", 
       y="Count",
       fill="Zip Code") +
  theme(plot.title = element_text(face="bold", hjust=0.5))

# Average price by zip code
avg_price <- aggregate(price ~ zip_code, house_data, mean)
p6 <- ggplot(avg_price, aes(x=zip_code, y=price, fill=zip_code)) +
  geom_col(alpha=0.7) +
  geom_text(aes(label=paste0("$", format(round(price/1000), big.mark=","), "K")), 
            vjust=-0.5, size=3.5) +
  scale_y_continuous(labels = scales::comma, limits=c(0, max(avg_price$price)*1.15)) +
  theme_minimal() +
  labs(title="Average Price by Zip Code", 
       x="Zip Code", 
       y="Average Price ($)") +
  theme(legend.position="none",
        plot.title = element_text(face="bold", hjust=0.5))

# Save EDA plots
png("visualizations/eda_plots.png", width=1600, height=1200, res=150)
grid.arrange(p1, p2, p3, p4, p5, p6, ncol=3, 
             top=grid::textGrob("Exploratory Data Analysis", 
                                gp=grid::gpar(fontsize=18, fontface="bold")))
dev.off()

cat("visualizations/eda_plots.png\n")

# ============================================================
# 2. CORRELATION HEATMAP
# ============================================================

cat("Creating correlation heatmap...\n")

# Create correlation matrix
cor_data <- house_data[, c('beds', 'baths', 'sqft', 'price')]
cor_matrix <- cor(cor_data)

# Melt for ggplot
cor_melted <- melt(cor_matrix)

# Create heatmap
png("visualizations/correlation_heatmap.png", width=1000, height=800, res=150)
ggplot(cor_melted, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile(color="white") +
  geom_text(aes(label=sprintf("%.2f", value)), color="black", size=5) +
  scale_fill_gradient2(low="blue", mid="white", high="red", 
                       midpoint=0, limits=c(-1,1)) +
  theme_minimal() +
  labs(title="Feature Correlation Matrix", 
       x="", y="", fill="Correlation") +
  theme(plot.title = element_text(face="bold", hjust=0.5, size=16),
        axis.text.x = element_text(angle=45, hjust=1, size=12),
        axis.text.y = element_text(size=12))
dev.off()

cat("visualizations/correlation_heatmap.png\n")

# ============================================================
# 3. CONFUSION MATRIX
# ============================================================

cat("Creating confusion matrix...\n")

# Create confusion matrix
cm <- table(Predicted=test_predictions$predicted, Actual=test_predictions$actual)

# Convert to data frame for plotting
cm_df <- as.data.frame(cm)

# Create heatmap
png("visualizations/confusion_matrix.png", width=900, height=800, res=150)
ggplot(cm_df, aes(x=Actual, y=Predicted, fill=Freq)) +
  geom_tile(color="white", size=1) +
  geom_text(aes(label=Freq), color="white", size=8, fontface="bold") +
  scale_fill_gradient(low="lightblue", high="darkblue") +
  theme_minimal() +
  labs(title="Confusion Matrix - Test Set Predictions", 
       x="True Label", 
       y="Predicted Label",
       fill="Count") +
  theme(plot.title = element_text(face="bold", hjust=0.5, size=16),
        axis.text = element_text(size=12),
        axis.title = element_text(size=14, face="bold"))
dev.off()

cat("visualizations/confusion_matrix.png\n")

# ============================================================
# 4. FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================

cat("Creating feature importance plot...\n")

# Reshape coefficients for plotting
coef_long <- melt(as.matrix(coefficients))
colnames(coef_long) <- c("ZipCode", "Feature", "Coefficient")

# Create grouped bar chart
png("visualizations/feature_importance.png", width=1200, height=800, res=150)
ggplot(coef_long, aes(x=Feature, y=Coefficient, fill=ZipCode)) +
  geom_bar(stat="identity", position="dodge", alpha=0.8) +
  geom_hline(yintercept=0, linetype="dashed", color="black") +
  theme_minimal() +
  labs(title="Logistic Regression Coefficients by Zip Code", 
       x="Feature", 
       y="Coefficient Value",
       fill="Zip Code") +
  theme(plot.title = element_text(face="bold", hjust=0.5, size=16),
        axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=11),
        axis.title = element_text(size=14, face="bold"),
        legend.position = "right")
dev.off()

cat("visualizations/feature_importance.png\n")

# ============================================================
# 5. MODEL PERFORMANCE METRICS
# ============================================================

cat("Creating performance metrics plot...\n")

# Calculate per-class metrics
zip_codes <- levels(test_predictions$actual)
metrics_list <- list()

for (zip in zip_codes) {
  tp <- sum(test_predictions$actual == zip & test_predictions$predicted == zip)
  fp <- sum(test_predictions$actual != zip & test_predictions$predicted == zip)
  fn <- sum(test_predictions$actual == zip & test_predictions$predicted != zip)
  
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  metrics_list[[zip]] <- data.frame(
    ZipCode = zip,
    Precision = precision,
    Recall = recall,
    F1Score = f1
  )
}

metrics_df <- do.call(rbind, metrics_list)
metrics_long <- melt(metrics_df, id.vars="ZipCode")

png("visualizations/performance_metrics.png", width=1000, height=700, res=150)
ggplot(metrics_long, aes(x=ZipCode, y=value, fill=variable)) +
  geom_bar(stat="identity", position="dodge", alpha=0.8) +
  geom_text(aes(label=sprintf("%.2f", value)), 
            position=position_dodge(width=0.9), 
            vjust=-0.5, size=3) +
  scale_y_continuous(limits=c(0, 1.1), breaks=seq(0, 1, 0.2)) +
  theme_minimal() +
  labs(title="Per-Class Performance Metrics", 
       x="Zip Code", 
       y="Score",
       fill="Metric") +
  theme(plot.title = element_text(face="bold", hjust=0.5, size=16),
        axis.text = element_text(size=12),
        axis.title = element_text(size=14, face="bold"))
dev.off()

cat("visualizations/performance_metrics.png\n")

# ============================================================
# 6. PRICE VS SIZE ANALYSIS
# ============================================================

cat("Creating price per sqft analysis...\n")

# Calculate price per sqft
house_data$price_per_sqft <- house_data$price / house_data$sqft

# Box plot
p1 <- ggplot(house_data, aes(x=zip_code, y=price_per_sqft, fill=zip_code)) +
  geom_boxplot(alpha=0.7) +
  geom_jitter(alpha=0.2, width=0.2) +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal() +
  labs(title="Price per Square Foot by Zip Code", 
       x="Zip Code", 
       y="Price per Sq Ft") +
  theme(legend.position="none",
        plot.title = element_text(face="bold", hjust=0.5))

# Violin plot
p2 <- ggplot(house_data, aes(x=zip_code, y=price_per_sqft, fill=zip_code)) +
  geom_violin(alpha=0.7, trim=FALSE) +
  geom_boxplot(width=0.1, alpha=0.9, outlier.shape=NA) +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal() +
  labs(title="Price per Sq Ft Distribution (Violin Plot)", 
       x="Zip Code", 
       y="Price per Sq Ft") +
  theme(legend.position="none",
        plot.title = element_text(face="bold", hjust=0.5))

png("visualizations/price_per_sqft.png", width=1200, height=600, res=150)
grid.arrange(p1, p2, ncol=2,
             top=grid::textGrob("Price per Square Foot Analysis", 
                                gp=grid::gpar(fontsize=16, fontface="bold")))
dev.off()

cat("visualizations/price_per_sqft.png\n")

# ============================================================
# SUMMARY
# ============================================================

cat("\n============================================================\n")
cat("R VISUALIZATION COMPLETE\n")
cat("============================================================\n\n")

cat("Generated visualizations in visualizations/:\n")
cat("eda_plots.png - Exploratory data analysis (6 plots)\n")
cat("correlation_heatmap.png - Feature correlations\n")
cat("confusion_matrix.png - Model predictions\n")
cat("feature_importance.png - Coefficient analysis\n")
cat("performance_metrics.png - Precision, Recall, F1\n")
cat("price_per_sqft.png - Price efficiency analysis\n")

cat("\nAll visualizations created successfully!\n")
cat("data/ for datasets and model outputs\n")
cat("visualizations/ for all plots\n")