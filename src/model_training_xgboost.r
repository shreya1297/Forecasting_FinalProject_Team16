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

train_xgb_model <- function(params) {

  params$objective <- "reg:squarederror"

  xgb.train(
    data = train_matrix,
    nrounds = 300,
    params = params,
    verbose = 0
  )
}

# Default model (used if no tuning)
default_params <- list(
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- train_xgb_model(default_params)
saveRDS(xgb_model, "data/processed/xgb_model_final.rds")
