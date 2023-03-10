import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from Helpers.features import calculate_prediction_sales

df = pd.read_csv("caspecoHistoricalDataProcessed.csv")

fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
fig.tight_layout()

df[["Sales"]] = df.apply(lambda row: calculate_prediction_sales(row, columns = ["SalesScaled"]), axis = "columns",  result_type='expand')

# AUTO CORRELATION PLOT
for i in range(3):
    ax = fig.add_subplot(3, 1, i + 1)
    ax.set_title(f"Weekly sales company {i}")
    ax.ticklabel_format(style = "plain")

    ax.plot(df[df["Company"] == i].groupby("Weekday")["Sales"].mean())

# plt.show()
plt.savefig("output/weekly.png", bbox_inches='tight')