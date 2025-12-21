# =====================================================
# Load libraries 
# =====================================================

library(dplyr)
library(tidyr)
library(lubridate)
library(slider)
library(forecast)
library(xgboost)
library(ggplot2)

source("src/model_training_xgboost.r")

## Load data
sales_long <- readRDS("data/processed/sales_long.rds")


# =====================================================
# Creating validation and evaluation datasets
# =====================================================

last_train_date <- max(sales_long$date)

validation_dates <- seq.Date(
  from = last_train_date + 1,
  by   = "day",
  length.out = 28
)

evaluation_dates <- seq.Date(
  from = last_train_date + 29,
  by   = "day",
  length.out = 28
)

# =====================================================
# Create future dataset — VALIDATION horizon (d_1914–d_1941)
# =====================================================

validation_future <- sales_long %>%
  distinct(item_id) %>%
  crossing(date = validation_dates) %>%
  left_join(
    sales_long %>% select(item_id, date, sell_price, event_flag, snap_TX),
    by = c("item_id", "date")
  )

combined_val <- bind_rows(
  sales_long,
  validation_future %>% mutate(sales = NA_real_)
)

combined_val <- combined_val %>%
  group_by(item_id) %>%
  arrange(date) %>%
  mutate(
    lag_28 = lag(sales, 28),
    lag_35 = lag(sales, 35),
    lag_42 = lag(sales, 42),
    rolling_mean_7  = slide_dbl(lag_28, mean, .before = 6,  .complete = TRUE),
    rolling_mean_28 = slide_dbl(lag_28, mean, .before = 27, .complete = TRUE),
    rolling_std_7   = slide_dbl(lag_28, sd,   .before = 6,  .complete = TRUE)
  ) %>%
  ungroup()

val_model_data <- combined_val %>%
  filter(date %in% validation_dates)

# =====================================================
# Read validation data
# =====================================================

val_truth <- read.csv("data/sales_test_validation_afcs2025.csv") %>%
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "d",
    values_to = "actual_sales"
  ) %>%
  mutate(item_id = id) %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup()

val_results <- val_model_data %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  left_join(val_truth, by = c("item_id", "h"))


# =====================================================
# Validation-based hyperparameter tuning
# =====================================================

param_grid <- expand.grid(
  eta = c(0.03, 0.05),
  max_depth = c(4, 5, 6),
  subsample = c(0.7, 0.8),
  colsample_bytree = c(0.7, 0.8)
)

tuning_results <- list()

for (i in seq_len(nrow(param_grid))) {

  params <- as.list(param_grid[i, ])

  cat("Training model", i, "of", nrow(param_grid), "\n")

  model <- train_xgb_model(params)

  val_matrix <- xgb.DMatrix(
    data = as.matrix(val_model_data[, feature_cols])
  )

  preds <- predict(model, val_matrix)

  rmse <- accuracy(
    preds,
    val_results$actual_sales
  )[,"RMSE"]

  tuning_results[[i]] <- cbind(param_grid[i, ], RMSE = rmse)

  cat("Validation RMSE:", rmse, "\n\n")
}

tuning_df <- bind_rows(tuning_results) %>%
  arrange(RMSE)

print(tuning_df)

best_params <- tuning_df[1, ]

# =====================================================
# Train FINAL model using best validation parameters
# =====================================================

best_params_clean <- best_params %>%
  select(-RMSE) %>%
  as.list()

xgb_model <- train_xgb_model(best_params_clean)
saveRDS(xgb_model, "data/processed/xgb_model_final.rds")

# =====================================================
# Validation RMSE for FINAL tuned model
# =====================================================

val_matrix <- xgb.DMatrix(
  data = as.matrix(val_model_data[, feature_cols])
)

val_model_data$pred_sales <- predict(
  xgb_model,
  val_matrix
)

val_results <- val_model_data %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  left_join(val_truth, by = c("item_id", "h"))

val_rmse <- accuracy(
  val_results$pred_sales,
  val_results$actual_sales
)[,"RMSE"]

val_rmse

# =====================================================
# Create future dataset — EVALUATION horizon (d_1942–d_1969)
# =====================================================

evaluation_future <- sales_long %>%
  distinct(item_id) %>%
  crossing(date = evaluation_dates) %>%
  left_join(
    sales_long %>% select(item_id, date, sell_price, event_flag, snap_TX),
    by = c("item_id", "date")
  )

combined_eval <- bind_rows(
  sales_long,
  evaluation_future %>% mutate(sales = NA_real_)
)

combined_eval <- combined_eval %>%
  group_by(item_id) %>%
  arrange(date) %>%
  mutate(
    lag_28 = lag(sales, 28),
    lag_35 = lag(sales, 35),
    lag_42 = lag(sales, 42),
    rolling_mean_7  = slide_dbl(lag_28, mean, .before = 6,  .complete = TRUE),
    rolling_mean_28 = slide_dbl(lag_28, mean, .before = 27, .complete = TRUE),
    rolling_std_7   = slide_dbl(lag_28, sd,   .before = 6,  .complete = TRUE)
  ) %>%
  ungroup()

eval_model_data <- combined_eval %>%
  filter(date %in% evaluation_dates)

# Predict on evaluation horizon
eval_matrix <- xgb.DMatrix(
  data = as.matrix(eval_model_data[, feature_cols])
)

eval_model_data$pred_sales <- predict(
  xgb_model,
  eval_matrix
)

# =====================================================
# Final RMSE — evaluation file (REPORTED RESULT)
# =====================================================

eval_truth <- read.csv("data/sales_test_evaluation_afcs_2025.csv") %>%
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "d",
    values_to = "actual_sales"
  ) %>%
  mutate(item_id = id) %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup()


eval_results <- eval_model_data %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  left_join(eval_truth, by = c("item_id", "h"))

final_rmse <- accuracy(
  eval_results$pred_sales,
  eval_results$actual_sales
)[,"RMSE"]

final_rmse

# =====================================================
# Plots
# =====================================================

# Validation: Actual vs Predicted (sample items)
val_results %>%
  filter(item_id %in% sample(unique(item_id), 2)) %>%
  ggplot(aes(x = h)) +
  geom_line(aes(y = actual_sales), color = "black", linewidth = 0.8) +
  geom_line(aes(y = pred_sales), color = "red", linewidth = 0.8) +
  facet_wrap(~ item_id, scales = "free_y") +
  labs(
    title = "Validation Forecasts: Actual vs Predicted Sales",
    x = "Forecast Horizon (days)",
    y = "Daily Sales"
  )

# Validation Plot — Actual vs Predicted (Selected Items)
items_to_plot <- c(
  "FOODS_3_356_TX_3_validation",
  "FOODS_3_730_TX_3_validation"
)

val_results %>%
  filter(item_id %in% items_to_plot) %>%
  ggplot(aes(x = h)) +
  geom_line(
    aes(y = actual_sales, color = "Actual"),
    linewidth = 0.8
  ) +
  geom_line(
    aes(y = pred_sales, color = "Predicted"),
    linewidth = 0.8
  ) +
  facet_wrap(~ item_id, scales = "free_y") +
  scale_color_manual(
    values = c("Actual" = "black", "Predicted" = "red"),
    name = ""
  ) +
  labs(
    title = "Validation Forecasts (28-Day Horizon): Actual vs Predicted Sales",
    x = "Forecast Horizon (Days)",
    y = "Daily Sales"
  ) + 
  theme(
    legend.position = c(0.98, 0.98),      # top-right inside
    legend.justification = c("right", "top"),
    legend.direction = "horizontal",
    legend.background = element_rect(
      fill = "white",
      color = NA
    ),
    legend.key = element_blank()
  )



# Validation: Error distribution
ggplot(val_results, aes(x = actual_sales - pred_sales)) +
  geom_histogram(
    bins = 60,
    fill = "#4C72B0",
    alpha = 0.85
  ) +
  coord_cartesian(xlim = c(-15, 15)) +
  labs(
    title = "Distribution of Forecast Errors (Validation)",
    x = "Prediction Error (Actual − Predicted)",
    y = "Count"
  )

# Evaluation (test): Actual vs Predicted (sample items)
eval_results %>%
  filter(item_id %in% sample(unique(item_id), 1)) %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  ggplot(aes(x = h)) +
  geom_line(aes(y = actual_sales), color = "black", linewidth = 0.8) +
  geom_line(aes(y = pred_sales), color = "red", linewidth = 0.8) +
  facet_wrap(~ item_id, scales = "free_y") +
  labs(
    title = "Evaluation Forecasts: Actual vs Predicted (Sample Items)",
    x = "Forecast Horizon (days)",
    y = "Daily Sales"
)

# Evaluation Plot — Actual vs Predicted (Selected Item)
eval_results %>%
  filter(grepl("FOODS_3_263", item_id)) %>%
  ggplot(aes(x = h)) +
  geom_line(
    aes(y = actual_sales, color = "Actual"),
    linewidth = 0.9
  ) +
  geom_line(
    aes(y = pred_sales, color = "Predicted"),
    linewidth = 0.9
  ) +
  scale_color_manual(
    values = c("Actual" = "black", "Predicted" = "red"),
    name = NULL
  ) +
  labs(
    title = "Evaluation Forecast: Actual vs Predicted Sales",
    subtitle = "Sample Item: FOODS_3_263_TX_3",
    x = "Forecast Horizon (Days)",
    y = "Daily Sales"
  ) +
  theme(
    legend.position = c(0.98, 0.98),
    legend.justification = c("right", "top"),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = "white", color = NA),
    legend.key = element_blank()
  )
