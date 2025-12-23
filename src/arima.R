library(data.table)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(slider)
library(fpp3)

# FORECASTING WITH ARIMA

# loading the dataset
sales_long <- readRDS("data/processed/sales_long.rds")

# 823 different items for FOOD 3
items <- unique(sales_long$item_id)

last_train_date <- max(sales_long$date)

# for each item fit a separate ARIMA model
# storing the forecasts for the validation set 
# storing the models found using search for each item time series
# storing the residuals for each fit 

validation_forecasts <- list()
evaluation_forecasts <- list()

arima_models_stepwise <- list()
residuals_list_stepwise <- list()

for (i in seq_along(items)){
  
  item <- items[i]
  
  data <- sales_long |>
    filter(item_id == item) |>
    as_tsibble(index = date)
  
  # fit ARIMA with seasonality search
  fit_arima <- data |>
    model(
      search = ARIMA(sales ~ PDQ() + pdq())
    )
  
  # forecasting validation and evaluation
  fc_arima <- forecast(fit_arima, h = 56)
  
  # store validation forecast results
  validation_forecasts[[i]] <- tibble(
    item_id = item,
    h = 1:28,
    pred_sales = as.numeric(fc_arima$.mean[1:28])
  )
  
  # store evaluation forecast results
  evaluation_forecasts[[i]] <- tibble(
    item_id = item,
    h = 1:28,
    pred_sales = as.numeric(fc_arima$.mean[29:56])
  )
  
  # store which model was used 
  arima_models_stepwise[[i]] <- glance(fit_arima) |> 
    mutate(item_id = item)
  
  # store residuals
  residuals_list_stepwise[[i]] <- augment(fit_arima) |>
    mutate(item_id = item)
  
}

val_predictions_search  <- bind_rows(validation_forecasts)
eval_predictions_search <- bind_rows(evaluation_forecasts)

arima_models     <- bind_rows(lapply(arima_models_stepwise, as_tibble))
arima_residuals  <- bind_rows(lapply(residuals_list_stepwise, as_tibble))

# Loading the ground truth for the validation set

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

# Joining the ground truth and predictions
val_results <- val_predictions_search %>%
  group_by(item_id) %>% 
  mutate(h = row_number()) %>%
  ungroup() %>%
  left_join(val_truth, by = c("item_id", "h"))

# Loading the ground truth for the evaluation set

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


eval_results <- eval_predictions_search %>%
  group_by(item_id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  left_join(eval_truth, by = c("item_id", "h"))

# Calculating the RMSE for the validation and evaluation set
val_rmse <- sqrt(mean((val_results$pred_sales - val_results$actual_sales)^2, 
                      na.rm = TRUE))

eval_rmse <- sqrt(mean((eval_results$pred_sales - eval_results$actual_sales)^2, 
                      na.rm = TRUE))

print(val_rmse)
print(eval_rmse)


# Evaluation Plot — Actual vs Predicted (Selected Item) - same code as for the plot in xgboost
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


## Analysing the chosen ARIMA models that were searched with 
library(purrr)

arima_orders <- arima_models_stepwise %>%
  map_df(~ tibble(
    item_id = .$item_id,
    p = length(.$ar_roots),
    d = .$model$d,          # if using fable ARIMA objects
    q = length(.$ma_roots)
  ))
arima_orders
