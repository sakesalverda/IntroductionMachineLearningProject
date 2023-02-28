import numpy as np
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("caspecoHistoricalDataProcessed.csv")
targetDf = pd.read_csv("caspecoTestRangeProcessed.csv")

dfObject = {
    0: df[df["Company"] == 0],
    1:  df[df["Company"] == 1],
    2:  df[df["Company"] == 2]
}

output_df = pd.read_csv("caspecoTestRange.csv")

for company in range(3):
    dfComp = dfObject[company]
    dfCompPredict = targetDf[targetDf["Company"] == company]

    x_train = dfComp.drop(["Sales", "Day", "Season"], axis = 1)
    y_train = dfComp["Sales"]

    x_predict = dfCompPredict.drop(["Day", "Season"], axis = 1)

    model = LinearRegression()
    model.fit(x_train, y_train)

    prediction = model.predict(x_predict)

    # We flip since x_predict is from first till last date, whereas the output_df is from last till first date
    output_df.loc[output_df["Company"] == company, "Sales"] = np.flip(prediction)


    # time_range = np.concatenate((y_train.tail(100).values, prediction))

    # plt.figure()
    # plt.plot(time_range)
    # plt.show()

    # print(prediction)

output_df["ID"] = output_df['Date'] + "_" + output_df['Company'].astype(str)
    

output_df.to_csv("Predictions/SimpleLinRegPrediction.csv", sep = ",", index = False, columns = ['ID', 'Sales'])
print(output_df.head(10))