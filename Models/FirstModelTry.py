import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("caspecoHistoricalDataProcessed.csv")

dfObject = {
    0: df[df["Company"] == 0],
    1:  df[df["Company"] == 1],
    2:  df[df["Company"] == 2]
}

for company in range(1):
    dfComp = dfObject[company]

    print(dfComp.head())


    