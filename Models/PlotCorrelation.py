import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("caspecoHistoricalDataProcessed.csv")

fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
fig.tight_layout()


# AUTO CORRELATION PLOT
from statsmodels.graphics.tsaplots import plot_acf
for i in range(3):
    ax = fig.add_subplot(3, 2, i * 2 + 1)
    ax2 = fig.add_subplot(3, 2, i * 2 + 2)
    ax.set_title(f"Auto correlation company {i}")
    ax.ticklabel_format(style = "plain")

    plot_acf(df[df["Company"] == i]["SalesScaled"], lags = 7*4, ax = ax)
    plot_pacf(df[df["Company"] == i]["SalesScaled"], lags = 7*4, ax = ax2)
# plt.show()
plt.savefig("output/autocorrelation.png")