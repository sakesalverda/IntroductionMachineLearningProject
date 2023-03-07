import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("caspecoHistoricalDataProcessed.csv")
targetDf = pd.read_csv("caspecoTestRangeProcessed.csv")

dfObject = {
    0: df[df["Company"] == 0],
    1:  df[df["Company"] == 1],
    2:  df[df["Company"] == 2]
}

output_df = pd.read_csv("caspecoTestRange_sorted.csv")

for company in range(1):
    dfComp = dfObject[company]
    dfCompPredict = targetDf[targetDf["Company"] == company]

    ### Split data

    # split without timeseries
    x_train = dfComp.drop(["Sales", "Day", "Season"], axis = 1)
    y_train = dfComp["Sales"]

    # split with timeseries
    splits = 5
    tscv = TimeSeriesSplit(n_splits = splits, test_size=int(len(x_train) / (splits + 1) - 1))
    for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
        print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")

        x_train_split, x_test_split = x_train.iloc[train_index, :], x_train.iloc[test_index,:]
        y_train_split, y_test_split = y_train.iloc[train_index], y_train.iloc[test_index]
        print(len(x_train_split))

        #Linear Regression

        #dt_reg = DecisionTreeRegressor(random_state=42)
        #dt_reg.fit(X=x_train_split, y=y_train_split)

        #dt_pred = dt_reg.predict(x_test_split)

        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(x_train_split, y=y_train_split.ravel())
        gbr_pred = gbr.predict(x_test_split)

        score = gbr.score(x_test_split, y_test_split)
        mse = mean_squared_error(y_test_split, gbr_pred)


        print(f"  Score: {score}")
        print(f"  MSE: {mse}")



