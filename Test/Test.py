import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Test/caspecoTrainingData.csv")

cols = "Sales"
g = df.groupby('Company')[cols]
min_, max_ = g.transform('min'), g.transform('max')
df[cols + '_scale'] = (df[cols] - min_) / (max_ - min_)

print(df.head())
# df["Sales"] = grouped["Sales"] - grouped["Sales"].min() / (grouped["Sales"].max() - grouped["Sales"].min())