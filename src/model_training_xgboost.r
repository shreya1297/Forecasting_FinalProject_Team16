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

## Load preprocessed data
sales_long <- readRDS("data/processed/sales_long.rds")
sales_long

# =====================================================
# Prepare data for modeling ---- check of needed!!
# =====================================================

# Ensure correct time ordering
sales_long <- sales_long %>%
  arrange(item_id, date)

# check of needed!!
# =====================================================
# Fill missing weekly prices across days (if any)
sales_long <- sales_long %>%
  group_by(item_id) %>%
  tidyr::fill(sell_price, .direction = "downup") %>%
  ungroup()

# =====================================================
# Feature Engineering: Create lag and rolling features
# =====================================================

# Testing criteria: LAG 7 / 14 / 28
# =====================================================

sales_long <- sales_long %>%
  group_by(item_id) %>%
  arrange(date) %>%
  mutate(
    # 1. LAG FEATURES
    # We use lag 28 because we need to predict 28 days ahead (Direct Strategy).
    # This prevents 'data leakage' where the model uses future info it won't have.
    lag_28 = lag(sales, 28),   # Sales exactly 4 weeks ago
    lag_35 = lag(sales, 35),   # Sales exactly 5 weeks ago (same weekday)
    lag_42 = lag(sales, 42),   # Sales exactly 6 weeks ago (same weekday)
    
    # 2. ROLLING WINDOW FEATURES (Moving Averages)
    # These smooth out daily noise and capture local trends/momentum.
    # Note: These are based on lag_28 to stay valid for the full forecast period.
    
    # Short-term trend: average sales over the week ending 28 days ago
    rolling_mean_7  = slide_dbl(lag_28, mean, .before = 6, .complete = TRUE),
    
    # Long-term baseline: average sales over the 4 weeks ending 28 days ago
    rolling_mean_28 = slide_dbl(lag_28, mean, .before = 27, .complete = TRUE),
    
    # Volatility: standard deviation to capture how 'unpredictable' the item is
    rolling_std_7   = slide_dbl(lag_28, sd, .before = 6, .complete = TRUE)
  ) %>%
  ungroup()


# =====================================================
# Remove rows with missing lag / rolling features
# =====================================================

sales_model <- sales_long %>%
  filter(
    !is.na(lag_42),
    !is.na(rolling_mean_28),
    !is.na(rolling_std_7)
  )

# =====================================================
# Feature selection - TESTING CRITERIA (wday, month to be added or not)
# =====================================================

feature_cols <- c(
  "lag_28", "lag_35", "lag_42",
  "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
  "sell_price", "event_flag", "snap_TX"
)

# =====================================================
# Prepare training matrix
# =====================================================
train_matrix <- xgb.DMatrix(
  data  = as.matrix(sales_model[, feature_cols]),
  label = sales_model$sales
)


# =====================================================
# Train XGBoost model
# =====================================================

set.seed(42)

xgb_model <- xgb.train(
  data = train_matrix,
  nrounds = 300,
  objective = "reg:squarederror",
  params = list(
    eta = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
)

saveRDS(xgb_model, "data/processed/xgb_model_final.rds")

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
# Predict on validation horizon (sanity check)
# =====================================================

val_matrix <- xgb.DMatrix(
  data = as.matrix(val_model_data[, feature_cols])
)

val_model_data$pred_sales <- predict(
  xgb_model,
  val_matrix
)

# Load validation truth
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

# Add horizon index to predictions
val_results <- val_model_data %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  left_join(val_truth, by = c("item_id", "h"))

summary(val_results$pred_sales)
summary(val_results$actual_sales)

# RMSE (sanity check only)
val_rmse <- accuracy(
  val_results$pred_sales,
  val_results$actual_sales
)[,"RMSE"]

val_rmse

val_results %>%
  filter(item_id %in% sample(unique(item_id), 3)) %>%
  ggplot(aes(x = h)) +
  geom_line(aes(y = actual_sales), color = "black") +
  geom_line(aes(y = pred_sales), color = "red") +
  facet_wrap(~ item_id, scales = "free_y") +
  labs(
    title = "Validation: Actual vs Predicted",
    y = "Sales",
    x = "Forecast Horizon"
  )


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
