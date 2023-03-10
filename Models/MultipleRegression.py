import numpy as np
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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
        model = LinearRegression()
        model.fit(x_train_split, y_train_split)

        prediction = model.predict(x_test_split)

        score = model.score(x_test_split, y_test_split)
        mse = mean_squared_error(y_test_split, prediction)

        print(f"  Score: {score}")
        print(f"  MSE: {mse}")




    # x_predict = dfCompPredict.drop(["Day", "Season"], axis = 1)

    # model = LinearRegression()
    # model.fit(x_train, y_train)

    # prediction = model.predict(x_predict)

    # output_df.loc[output_df["Company"] == company, "Sales"] = prediction


    # time_range = np.concatenate((y_train.tail(20).values, prediction))

    # plt.figure()
    # plt.plot(time_range)
    # plt.show()

    # print(prediction)

output_df["ID"] = output_df['Date'] + "_" + output_df['Company'].astype(str)
    

# output_df.to_csv("Predictions/SimpleLinRegPrediction.csv", sep = ",", index = False, columns = ['ID', 'Sales'])
print(output_df.head(10))