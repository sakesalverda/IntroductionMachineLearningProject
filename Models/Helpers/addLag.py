import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def addLag(df):
    # For the historical data there is sales data
    # for the prediction set there is no sales data
    if "Sales" in df.columns:
        df["Lag7"] = df["Sales"]
        df["Lag7"].shift(21) # 7*3

        df["Lag1"] = df["Sales"]
        df["Lag1"].shift(3)
    else:
        # @important / @todo
        # Due to the lag we should predict per day and not all days at once
        # when that day is predicted, the lag1 for the next day is just that prediction
        # and lag7 is the sales data for 7 days earlier, which could either be a prediction
        # or a sales in the historical data
        df["Lag7"] = 0
        df["Lag1"] = 0

    df.dropna() 

    return df
