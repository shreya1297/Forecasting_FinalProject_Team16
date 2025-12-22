
# Seasonal Naive (Weekly) baseline
# - Uses preprocessed data: data/processed/sales_long.rds
# - Computes RMSE on:
#   (1) Validation set: sales_test_validation_afcs2025.csv (28 days)
#   (2) Evaluation set: sales_test_evaluation_afcs_2025.csv (28 days)
# For evaluation horizon, we CANNOT use "date-7" because
#   those dates fall in the unknown validation period. Instead we
#   repeat the last observed week from training for all future days.

library(dplyr)
library(tidyr)
library(lubridate)


# Helpers
rmse <- function(pred, actual) {
  ok <- !is.na(pred) & !is.na(actual)
  if (sum(ok) == 0) return(NaN)
  sqrt(mean((pred[ok] - actual[ok])^2))
}

# Seasonal Naive: repeat last observed seasonal cycle (7 days)
# This is compatible with your setup because you only have training sales
# and you do NOT observe the intermediate 28 validation days when forecasting
# the evaluation horizon.
snaive_repeat_last_week <- function(df, future_dates, seasonal_period = 7) {
  stopifnot(all(c("item_id", "date", "sales") %in% names(df)))
  stopifnot(length(future_dates) > 0)
  
  df <- df %>% arrange(item_id, date)
  
  # last 7 observed values per item_id from training
  last_week_tbl <- df %>%
    group_by(item_id) %>%
    summarise(
      n_obs = dplyr::n(),
      last_week = list(tail(sales, seasonal_period)),
      .groups = "drop"
    )
  
  # Safety: ensure every item has at least 7 observations
  if (any(last_week_tbl$n_obs < seasonal_period)) {
    bad <- last_week_tbl %>% filter(n_obs < seasonal_period) %>% pull(item_id)
    stop(
      "Some item_id have fewer than ", seasonal_period, " observations. Example: ",
      paste(head(bad, 5), collapse = ", ")
    )
  }
  
  # Expand to future dates and repeat pattern
  last_week_tbl %>%
    select(item_id, last_week) %>%
    tidyr::crossing(date = future_dates) %>%
    group_by(item_id) %>%
    mutate(
      h = row_number(),
      pred_sales = last_week[[1]][((h - 1) %% seasonal_period) + 1]
    ) %>%
    ungroup() %>%
    select(item_id, date, h, pred_sales)
}


# Load preprocessed data
sales_long <- readRDS("data/processed/sales_long.rds") %>%
  arrange(item_id, date)

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

# Predictions
val_pred  <- snaive_repeat_last_week(sales_long, validation_dates, seasonal_period = 7)
eval_pred <- snaive_repeat_last_week(sales_long, evaluation_dates, seasonal_period = 7)

# Truth (reshape wide -> long)
# Note: evaluation ids end with _evaluation, while training ids end with _validation
# We align them by replacing suffix in evaluation truth.
val_truth <- read.csv("data/sales_test_validation_afcs2025.csv") %>%
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "d",
    values_to = "actual_sales"
  ) %>%
  mutate(item_id = id) %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  select(item_id, h, actual_sales)

eval_truth <- read.csv("data/sales_test_evaluation_afcs_2025.csv") %>%
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "d",
    values_to = "actual_sales"
  ) %>%
  mutate(
    item_id = gsub("_evaluation$", "_validation", id)  # align with training ids
  ) %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  select(item_id, h, actual_sales)


# Join + RMSE
val_join  <- val_pred  %>% left_join(val_truth,  by = c("item_id", "h"))
eval_join <- eval_pred %>% left_join(eval_truth, by = c("item_id", "h"))

# Guards 
stopifnot(sum(!is.na(val_join$pred_sales)) > 0)
stopifnot(sum(!is.na(eval_join$pred_sales)) > 0)
stopifnot(sum(!is.na(val_join$actual_sales)) > 0)
stopifnot(sum(!is.na(eval_join$actual_sales)) > 0)

val_rmse_snaive  <- rmse(val_join$pred_sales,  val_join$actual_sales)
eval_rmse_snaive <- rmse(eval_join$pred_sales, eval_join$actual_sales)

cat("Seasonal Naive (Weekly) Validation RMSE:", val_rmse_snaive, "\n")
cat("Seasonal Naive (Weekly) Evaluation RMSE:", eval_rmse_snaive, "\n")


# Optional quick diagnostic plot 
library(ggplot2)
items_to_plot <- sample(unique(val_join$item_id), 2)
val_join %>%
  filter(item_id %in% items_to_plot) %>%
  ggplot(aes(x = h)) +
  geom_line(aes(y = actual_sales, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = pred_sales, color = "Predicted"), linewidth = 0.8) +
  facet_wrap(~ item_id, scales = "free_y") +
  labs(
    title = "Seasonal Naive (Weekly) — Validation Forecasts: Actual vs Predicted",
    x = "Forecast Horizon (Days)",
    y = "Daily Sales"
  ) +
  theme(legend.title = element_blank())
