## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE,
  cache = FALSE,
  fig.width = 10,
  fig.height = 4,
  dev.args = list(pointsize = 11)
)

options(digits = 3, width = 60)


## -----------------------------------------------------------------------------
library(data.table)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(slider)


## -----------------------------------------------------------------------------
sales <- read.csv("data/sales_train_validation_afcs2025.csv")
calendar_data <- read.csv("data/calendar_afcs2025.csv")
prices <- read.csv("data/sell_prices_afcs2025.csv")


## -----------------------------------------------------------------------------
sales_long <- sales %>%
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "d",
    values_to = "sales"
  )

sales_long <- sales_long %>%
  separate(id, into = c("cat_id","item_num","store","dataset"), sep = "_", extra = "merge",
           remove = FALSE) %>%
  mutate(item_id = id)
sales_long


## -----------------------------------------------------------------------------
calendar <- calendar_data %>%
  mutate(d = paste0("d_", row_number()))


## -----------------------------------------------------------------------------
sales_long <- sales_long %>%
  left_join(calendar, by = "d")
sales_long %>%
  select(cat_id, item_num, store, dataset, d, date, weekday, sales) %>%
  head(10)
sales_long <- sales_long %>%
  mutate(date = as.Date(date, format = "%m/%d/%Y"))
sales_long


## -----------------------------------------------------------------------------
head(sales_long %>% select(d, date, sales, event_name_1))
summary(sales_long$sales)
table(sales_long$event_name_1)


## -----------------------------------------------------------------------------
sales_long <- sales_long %>%
  mutate(
    year  = year(date),
    month = month(date),
    week  = wm_yr_wk,
    wday  = wday(date, label = TRUE),      
    is_weekend = ifelse(wday %in% c("Sat", "Sun"), 1, 0),
    event_flag = ifelse(!is.na(event_name_1), 1, 0)
  )


## -----------------------------------------------------------------------------
sales_long <- sales_long %>%
  rename(store_id = store) %>%
  left_join(prices, by = c("store_id", "item_id", "wm_yr_wk"))


## -----------------------------------------------------------------------------
sales_long %>%
  filter(item_id == "FOODS_3_001") %>%
  ggplot(aes(date, sales)) +
  geom_line() +
  labs(title = "Daily Sales for FOODS_3_001")


## -----------------------------------------------------------------------------
sales_long %>%
  group_by(wday) %>%
  summarise(avg_sales = mean(sales)) %>%
  ggplot(aes(wday, avg_sales)) +
  geom_col(fill="skyblue") +
  labs(title="Average Sales by Weekday")


## -----------------------------------------------------------------------------
sales_long %>%
  filter(event_flag == 1) %>%
  ggplot(aes(date, sales)) +
  geom_point(color="red") +
  labs(title="Sales on Event Days")

## Persist the final preprocessed dataset to ensure reproducibility and
## consistent feature availability across forecasting models.
dir.create("data/processed", showWarnings = FALSE)
saveRDS(sales_long, "data/processed/sales_long.rds")


