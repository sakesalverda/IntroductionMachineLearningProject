# 0) Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Helpers.metrics import error_metrics
from Helpers.df import split_df, add_prediction_to_df, get_train_vars_df, one_hot_encode_df
from Helpers.model import plot

from sklearn.linear_model import LinearRegression

identifier = "Baseline"


# 1) Read the dataframe
df = pd.read_csv("caspecoHistoricalDataProcessed.csv")



# 2) Add features, that are not sensitive to data leakage

df = one_hot_encode_df(df)

# skip since this is the baseline model



# 3) Split the dataframe into traintest and validation
train_df, validation_df = split_df(df)



# 4) Create model and fit optimum hyper parameters on train_df
model = LinearRegression()
model.fit(get_train_vars_df(train_df), train_df["SalesScaled"])



# 5) Train on entire train_df using optimised hyper paramaters

# skip since this is the baseline model, we already trained on train_df



# 6) Predict on validation_df and reverse the minmax scaling
y_pred = model.predict(get_train_vars_df(validation_df))

add_prediction_to_df(validation_df, y_pred)



# 7) Calculate error metrics
metrics = error_metrics(validation_df)

print(metrics)



# 8) plot against actual, and perhaps baseline
plot(validation_df, name = identifier)


# 9) Use model to predict future set
