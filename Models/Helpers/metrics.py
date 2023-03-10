import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from Helpers.df import get_rows_for_company_df

def weighted_mean_absolute_percentage_error(y_true, y_pred): 
    y_true = y_true.values
    y_pred = y_pred.values

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) # * 100 # if we want it from 0-100

def error_metrics(df):
    prediction = df["Prediction"]
    true = df["Sales"]

    rmse = np.empty(4)
    wmape = np.empty(4)
    mape = np.empty(4)
    mae = np.empty(4)

    rmse[0] = mean_squared_error(df["Prediction"], df["Sales"], squared = False)
    wmape[0] = weighted_mean_absolute_percentage_error(df["Prediction"], df["Sales"])
    mape[0] = mean_absolute_percentage_error(df["Prediction"], df["Sales"])
    mae[0] = mean_absolute_error(df["Prediction"], df["Sales"])

    for i in range(3):
        rows = get_rows_for_company_df(df, i)
        rmse[i + 1] = mean_squared_error(df[rows]["Prediction"], df[df["Company"] == i]["Sales"], squared = False)
        wmape[i + 1] = weighted_mean_absolute_percentage_error(df[rows]["Prediction"], df[df["Company"] == i]["Sales"])
        mape[i + 1] = mean_absolute_percentage_error(df[rows]["Prediction"], df[df["Company"] == i]["Sales"])
        mae[i + 1] = mean_absolute_error(df[rows]["Prediction"], df[df["Company"] == i]["Sales"])

    df = pd.DataFrame(data = {
        "Company": ["All", 0, 1, 2],
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape * 100,
        "WMAPE": wmape * 100
    })

    df = df.round(2)

    return df
    