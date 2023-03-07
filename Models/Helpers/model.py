import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

from Helpers.df import get_train_vars_df
from Helpers.df import test_size
from Helpers.df import calculate_prediction_sales

simple_seasons = [
    len(pd.date_range(start = "2020-01-01", end = "2020-03-21")),
    len(pd.date_range(start = "2020-03-22", end = "2020-06-21")),
    len(pd.date_range(start = "2020-06-22", end = "2020-09-21")),
    len(pd.date_range(start = "2020-09-22", end = "2020-12-21")),
    len(pd.date_range(start = "2020-12-22", end = "2021-03-21")),
    len(pd.date_range(start = "2021-03-22", end = "2021-06-21")),
    len(pd.date_range(start = "2021-06-22", end = "2021-09-21")),
    len(pd.date_range(start = "2021-09-22", end = "2021-12-21")),
    len(pd.date_range(start = "2021-12-22", end = "2022-03-21")),
    len(pd.date_range(start = "2022-03-22", end = "2022-06-21")),
    len(pd.date_range(start = "2022-06-22", end = "2022-09-21")),
    len(pd.date_range(start = "2022-09-22", end = "2022-12-21")),
    len(pd.date_range(start = "2022-12-22", end = "2023-01-04"))
]

def tunecv(train_x, train_y, model, param_grid):
    tscv = TimeSeriesSplit(n_splits = 5, test_size = test_size)
    
    CV_gs = GridSearchCV(estimator = model, param_grid = param_grid, cv = tscv, verbose = 2)
    CV_gs.fit(train_x, train_y)

    return CV_gs

def plot(df, name):
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.tight_layout()

    seasons = simple_seasons

    for i in range(3):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.set_title(f"Sales company {i}")
        ax.ticklabel_format(style = "plain")
        
        sales = df[df["Company"] == i]["Sales"]
        prediction = df[df["Company"] == i]["Prediction"]
        
        line_color = ""
        if i == 1:
            line_color = "darkorange"
        elif i == 2:
            line_color = "green"

        ax.plot(range(len(sales)), sales, "black", alpha = 0.3, linestyle = "dashed", label = "Actual")
        ax.plot(range(len(sales)), prediction, line_color, label = "Predicted")

        ax.legend()
        # use this line to compare with these since salary, it can be seen this indeed has an influence on sales
        #ax.plot(range(len(sales)), df[df["Company"] == i]["DaysSinceSalary"] / 31 * max_sales * 0.2)

    # plt.show()
    plt.savefig(f"Output/ValidationGraphs/{name}", bbox_inches='tight')



def forecast(model, name):
    df = pd.read_csv("caspecoHistoricalDataProcessed.csv")
    df_predict = pd.read_csv("caspecoTestRangeProcessed.csv")

    df["SalesScaledLastDay"] = df.groupby("Company")["SalesScaled"].shift(1)
    df["SalesScaledLastWeek"] = df.groupby("Company")["SalesScaled"].shift(7)
    df["SalesRollingMeanWeek"] = df.groupby("Company")["SalesScaled"].shift(1).rolling(7).mean().reset_index(0, drop = True)
    df["SalesRollingMean2Week"] = df.groupby("Company")["SalesScaled"].shift(1).rolling(14).mean().reset_index(0, drop = True)

    df = df.tail(n = 21*3)

    df = pd.concat([df, df_predict])

    # print(df)

    for i in range(15):
        # predict 15 times
        for company in range(3):
            row = df[df["Company"] == company].head(21 + i + 1).tail(1)
            # row = df.loc[(df["Date"] == row["Date"].values[0]) & (df["Company"] == row["Company"].values[0]), :]

            row["SalesScaledLastDay"] = df[df["Company"] == company].head(21 + i + 1 - 1).tail(1)["SalesScaled"].values
            row["SalesScaledLastWeek"] = df[df["Company"] == company].head(21 + i + 1 - 7).tail(1)["SalesScaled"].values
            row["SalesRollingMeanWeek"] = df[df["Company"] == company].head(21 + i + 1 - 7).tail(7)["SalesScaled"].mean()
            row["SalesRollingMean2Week"] = df[df["Company"] == company].head(21 + i + 1 - 14).tail(14)["SalesScaled"].mean()

            y_pred = model.predict(get_train_vars_df(row))[0]

            # row["SalesScaled"] = y_pred
            # df.iloc[21*3 + i*3 + company, "SalesScaled"] = y_pred
            df.loc[(df["Date"] == row["Date"].values[0]) & (df["Company"] == row["Company"].values[0]), "SalesScaled"] = y_pred
            # row["SalesScaled"] = y_pred
    
    predicted_df = df.tail(15*3).copy()

    # predicted_df.insert(len(predicted_df.columns), "Sales", predicted_df.apply(lambda row: calculate_prediction_sales(row, columns=["SalesScaled"]), axis = "columns",  result_type='expand'))

    predicted_df["Sales"] = predicted_df.apply(lambda row: calculate_prediction_sales(row, columns=["SalesScaled"]), axis = "columns",  result_type='expand')
    predicted_df["ID"] = predicted_df["Date"] + "_" + predicted_df["Company"].astype(str)

    print(predicted_df["ID"])
    predicted_df.to_csv(f"Output/Forecast/CSV/{name}.csv", sep = ",", index = False, columns = ['ID', 'Sales'])

    # print(predicted_df.tail(15*3)["Sales"])