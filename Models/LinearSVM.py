# 0) Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Helpers.metrics import error_metrics
from Helpers.df import split_df, add_prediction_to_df, get_train_vars_df, one_hot_encode_df
from Helpers.model import tunecv, forecast, plot, add_lag_features

from sklearn.svm import LinearSVR as Regressor

identifier = "LinearSVM"
linearModel = True


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
# param_grid = {"kernel": ["rbf", "poly"], "gamma": ["auto", "scale"], "degree": range(1, 6, 1)}
# param_grid = {"max_iter": [30000], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# rfr = Regressor(random_state = 8)
# cv_rfr = tunecv(get_train_vars_df(train_df), train_df["SalesScaled"], rfr, param_grid = param_grid)

# print(cv_rfr.best_params_)
# # output: 

# exit() # use during hyper paramater testing

# 5) Train on entire train_df using optimised hyper paramaters

# best_params_grid = {"max_depth": 9, "max_features": 0.3, "n_estimators": 500}
best_params_grid = {'learning_rate': 0.01, 'n_estimators': 2800}
best_params_grid = {"max_iter": 50000, "C": 0.01}
model = Regressor(**best_params_grid, random_state = 8)
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
model = Regressor(**best_params_grid, random_state=8)
model.fit(get_train_vars_df(df), df["SalesScaled"])
forecast(model, name = identifier, one_hot = linearModel)