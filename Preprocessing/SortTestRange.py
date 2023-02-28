import pandas as pd

predict_set = "caspecoTestRange.csv"

predict_df = pd.read_csv(predict_set)

predict_df.sort_values(by = ["Date", "Company"], inplace = True)

predict_df.to_csv("caspecoTestRange_sorted.csv", sep = ",", index = False)
