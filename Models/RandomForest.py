# 0) Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Helpers.metrics import error_metrics
from Helpers.df import split_df, add_prediction_to_df, get_train_vars_df
from Helpers.model import tunecv, forecast, plot, add_lag_features

from sklearn.ensemble import RandomForestRegressor


# 1) Read the dataframe
df = pd.read_csv("caspecoHistoricalDataProcessed.csv")


# 2) Add features, that are not sensitive to data leakage

add_lag_features(df)

df.dropna(inplace=True)



# 3) Split the dataframe into traintest and validation
train_df, validation_df = split_df(df)



# 4) Optimise hyperparamaters
# param_grid = {'max_depth' : [3, 5, 7]}
# param_grid = {'n_estimators': np.arange(50, 200, 15),
#               'max_features': np.arange(0.1, 1, 0.1),
#               'max_depth': [3, 5, 7, 9],
#               'max_samples': [0.3, 0.5, 0.8]}

# rfr = RandomForestRegressor(random_state = 8)
# cv_rfr = tunecv(get_train_vars_df(train_df), train_df["SalesScaled"], rfr, param_grid = param_grid)

# print(cv_rfr.best_params_)
# output: {'max_depth': 9, 'max_features': 0.5, 'max_samples': 0.8, 'n_estimators': 50}

# exit() # use during hyper paramater testing

# 5) Train on entire train_df using optimised hyper paramaters

# best_params_grid = {"max_depth": 9, "max_features": 0.3, "n_estimators": 500}
best_params_grid = {'max_depth': 9, 'max_features': 0.5, 'max_samples': 0.8, 'n_estimators': 50}
model = RandomForestRegressor(**best_params_grid, random_state = 8)
model.fit(get_train_vars_df(train_df), train_df["SalesScaled"])



# 6) Predict on validation set and reverse the minmax scaling
y_pred = model.predict(get_train_vars_df(validation_df))

add_prediction_to_df(validation_df, y_pred)



# 7) Calculate error metrics
metrics = error_metrics(validation_df)

print(metrics)



# 8) plot against actual, and perhaps baseline
plot(validation_df, name = "RandomForest")


# 9) Use model to predict future set
model = RandomForestRegressor(**best_params_grid, random_state = 8)
model.fit(get_train_vars_df(df), df["SalesScaled"])
forecast(model, name = "RandomForest")