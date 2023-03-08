# 0) Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Helpers.metrics import error_metrics
from Helpers.df import split_df, add_prediction_to_df, get_train_vars_df, one_hot_encode_df
from Helpers.model import tunecv, forecast, plot, add_lag_features

from sklearn.neighbors import KNeighborsRegressor as Regressor

identifier = "KNeighbors"
linearModel = False


# 1) Read the dataframe
df = pd.read_csv("caspecoHistoricalDataProcessed.csv")


# 2) Add features, that are not sensitive to data leakage

if linearModel == True:
    df = one_hot_encode_df(df)

add_lag_features(df)

df.dropna(inplace=True)



# 3) Split the dataframe into traintest and validation
train_df, validation_df = split_df(df)



# 4) Optimise hyperparamaters
# param_grid = {'n_neighbors': [3, 5, 11, 19], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}

# rfr = Regressor()
# cv_rfr = tunecv(get_train_vars_df(train_df), train_df["SalesScaled"], rfr, param_grid = param_grid)

# print(cv_rfr.best_params_)
# # output: {'metric': 'manhattan', 'n_neighbors': 19, 'weights': 'distance'}

# exit() # use during hyper paramater testing

# 5) Train on entire train_df using optimised hyper paramaters

# best_params_grid = {"max_depth": 9, "max_features": 0.3, "n_estimators": 500}
# best_params_grid = {'colsample_bytree': 0.8, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 600, 'subsample': 0.8}
best_params_grid = {'metric': 'manhattan', 'n_neighbors': 19, 'weights': 'distance'}
model = Regressor(**best_params_grid)
model.fit(get_train_vars_df(train_df), train_df["SalesScaled"])



# 6) Predict on validation set and reverse the minmax scaling
y_pred = model.predict(get_train_vars_df(validation_df))

add_prediction_to_df(validation_df, y_pred)



# 7) Calculate error metrics
metrics = error_metrics(validation_df)

print(metrics)



# 8) plot against actual, and perhaps baseline
plot(validation_df, name = identifier)


# 9) Use model to predict future set
model = Regressor(**best_params_grid)
model.fit(get_train_vars_df(df), df["SalesScaled"])
forecast(model, name = identifier, one_hot = linearModel)