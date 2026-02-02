# FULL MODELING PIPELINE: Stepwise Regression, Random Forest & GBM
# ──────────────────────────────────────────────────────────────────────────────
# 0) Libraries
library(tidymodels)    # recipes, rsample, etc.
library(tidyverse)     # readr, dplyr, stringr, etc.
library(randomForest)  # random forests
library(gbm)           # gradient boosting machines
# 1) Load the data
austinhouses <- read_csv("austinhouses.csv")
# 2) Feature engineering
austinhouses <- austinhouses %>%
  mutate(
    has_pool       = str_detect(tolower(description), "pool"),
    has_renovation = str_detect(tolower(description), "renovated|renovation"),
    age            = latest_saleyear - yearBuilt
  ) %>%
  # Drop raw columns already encoded above and any date column
  select(-yearBuilt,
         -description,
         -streetAddress,
         -latest_salemonth,
         -latest_saleyear,
         -latest_saledate)
# 3) Convert categorical & logical flags to factors
austinhouses <- austinhouses %>%
  mutate(
    zipcode        = factor(zipcode),
    hasAssociation = factor(hasAssociation),
    hasSpa         = factor(hasSpa),
    hasView        = factor(hasView),
    homeType       = factor(homeType)
  ) %>%
  # Turn logical features (pool, renovation) into factors
  mutate(across(
    .cols = where(is.logical),
    .fns  = ~ factor(.x, levels = c(FALSE, TRUE), labels = c("no","yes"))
  ))
# 4) Preprocessing recipe: drop zero-variance columns & dummy-encode
recipe_house <-
  recipe(latestPrice ~ ., data = austinhouses) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE)
prep_recipe      <- prep(recipe_house)
data_processed   <- bake(prep_recipe, new_data = NULL)
# 5) Train/test split (80/20 stratified on price)
set.seed(123)
split_data <- initial_split(data_processed, prop = 0.8, strata = latestPrice)
train_data <- training(split_data)
test_data  <- testing(split_data)
# 6) Stepwise regression (explicitly call stats::step)
set.seed(123)
full_lm    <- lm(latestPrice ~ ., data = train_data)
step_model <- stats::step(
  object    = full_lm,
  direction = "both",
  scope     = formula(full_lm)
)
step_preds <- predict(step_model, newdata = test_data)
rmse_step <- sqrt(mean((test_data$latestPrice - step_preds)^2))
mae_step  <- mean(abs(test_data$latestPrice - step_preds))
cat(":chart_with_downwards_trend: Stepwise Regression — RMSE:", round(rmse_step, 2),
    " MAE:", round(mae_step, 2), "\n\n")
# 7) Random Forest
p <- ncol(train_data) - 1
set.seed(123)
rf_mod    <- randomForest(
  latestPrice ~ .,
  data       = train_data,
  mtry       = floor(sqrt(p)),  # integer
  ntree      = 500,
  importance = TRUE
)
rf_preds  <- predict(rf_mod, newdata = test_data)
rmse_rf <- sqrt(mean((test_data$latestPrice - rf_preds)^2))
mae_rf  <- mean(abs(test_data$latestPrice - rf_preds))
cat(":chart_with_downwards_trend: Random Forest — RMSE:", round(rmse_rf, 2),
    " MAE:", round(mae_rf, 2), "\n\n")
# 8) Gradient Boosting Machine (gbm)
set.seed(123)
gbm_mod <- gbm(
  formula           = latestPrice ~ .,
  data              = train_data,
  distribution      = "gaussian",
  n.trees           = 5000,
  interaction.depth = 4,
  shrinkage         = 0.01,
  n.minobsinnode    = 10,
  cv.folds          = 5,
  verbose           = FALSE
)
best_trees <- gbm.perf(gbm_mod, method = "cv")
cat("Optimal # of trees for GBM:", best_trees, "\n")
gbm_preds <- predict(gbm_mod, newdata = test_data, n.trees = best_trees)
rmse_gbm <- sqrt(mean((test_data$latestPrice - gbm_preds)^2))
mae_gbm  <- mean(abs(test_data$latestPrice - gbm_preds))
cat(":chart_with_downwards_trend: GBM — RMSE:", round(rmse_gbm, 2),
    " MAE:", round(mae_gbm, 2), "\n\n")
# 9) Compare all methods
results <- tibble(
  Model = c("Stepwise", "Random Forest", "GBM"),
  RMSE  = c(rmse_step, rmse_rf, rmse_gbm),
  MAE   = c(mae_step, mae_rf, mae_gbm)
)
print(results)



# ───────────────────────────────────────────────────────────────






# R-squared for Random Forest
rsq_rf <- 1 - sum((y_test - rf_preds)^2) / sum((y_test - mean(y_test))^2)
cat("R-squared (Random Forest):", round(rsq_rf, 4), "\n")

# Our model explains 78.6% of the variability in house prices 
# that’s strong performance for real-world data like housing, which is naturally noisy and complex.

# ───────────────────────────────────────────────────────────────────────

# Residuals for Random Forest
rf_residuals <- y_test - rf_preds

# Plot
plot(rf_preds, rf_residuals,
     main = "Residuals vs Predictions (RF)",
     xlab = "Predicted Price",
     ylab = "Residuals",
     pch = 20, col = "blue")
abline(h = 0, col = "red")

# Centered around zero:
# Most residuals are centered around the red horizontal line (zero), 
# which means your model isn't consistently over- or under-predicting.
# 
# No obvious curve/trend:
# There’s no strong non-linear pattern, 
# which means your model captures most of the structure in the data.
# 
# Heteroscedasticity is mild:
# While the spread of residuals increases slightly as the predicted price increases, 
# it’s not alarming — common in real estate data where high-value homes are harder to predict accurately

# ───────────────────────────────────────────────────────────────────────--

library(caret)

set.seed(123)

# For Random Forest
cv_rf <- train(
  latestPrice ~ ., data = austin_train,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 3
)
print(cv_rf)

# ──────────────────────────────────────────────────────────────

# Train RMSE vs Test RMSE
# ---------------------------
# Predict on train set
rf_train_preds <- predict(rf_model, newdata = austin_train)

# Train RMSE
rf_train_rmse <- sqrt(mean((austin_train$latestPrice - rf_train_preds)^2))

# Test RMSE
rf_test_rmse <- sqrt(mean((y_test - rf_preds)^2))

cat("RF Train RMSE: $", round(rf_train_rmse, 2), "\n")
cat("RF Test RMSE: $", round(rf_test_rmse, 2), "\n")

# ────────────────────────────────────────────────────────

readr::write_csv(
    tibble(ID = seq_len(nrow(test_data)), PredictedPrice = rf_preds),
    "pred_random_forest_test.csv"
  )
# ────────────────────────────────────────────────────────
# 1. Read the holdout file
  holdout <- read_csv("austinhouses_holdout.csv")

# 2. Apply the same feature engineering
holdout <- holdout %>%
  mutate(
    has_pool       = str_detect(tolower(description), "pool"),
    has_renovation = str_detect(tolower(description), "renovated|renovation"),
    age            = latest_saleyear - yearBuilt
  ) %>%
  select(-yearBuilt,
         -description,
         -streetAddress,
         -latest_salemonth,
         -latest_saleyear,
         -latest_saledate) %>%
  mutate(
    zipcode        = factor(zipcode),
    hasAssociation = factor(hasAssociation),
    hasSpa         = factor(hasSpa),
    hasView        = factor(hasView),
    homeType       = factor(homeType)
  ) %>%
  mutate(across(
    .cols = where(is.logical),
    .fns  = ~ factor(.x, levels = c(FALSE, TRUE), labels = c("no", "yes"))
  ))

# 3. Preprocess using the same recipe as training
holdout_processed <- bake(prep_recipe, new_data = holdout)

# 4. Align columns with training data
missing_cols <- setdiff(names(train_data), names(holdout_processed))
for (col in missing_cols) {
  holdout_processed[[col]] <- 0
}
holdout_processed <- holdout_processed[, names(train_data)]

# 5. Predict with the trained Random Forest
holdout_preds <- predict(rf_mod, newdata = holdout_processed)

# 6. Create final submission file
submission <- tibble(
  ID = holdout$ID,  # assumes holdout has an ID column
  PredictedPrice = holdout_preds
)

# 7. Save to CSV
write_csv(submission, "pred_random_forest_holdout.csv")











