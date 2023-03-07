import numpy as np
import pandas as pd
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error # mean_squared_error actually is root mean squared error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

def weighted_mean_absolute_percentage_error(y_true, y_pred): 
    # y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) # * 100 # if we want it from 0-100

df = pd.read_csv("caspecoHistoricalDataProcessed.csv")

# about 157 weeks in total
# minus 4 for validation (we choose)
#
# 153 weeks for traintest
#
# split 5
# 30 weeks per split
# 4 weeks
# 26 weeks for training

df["SalesScaledLastDay"] = df["SalesScaled"].shift(1*3)
df["SalesScaledLastWeek"] = df["SalesScaled"].shift(7*3)

# df["SalesRollingMeanWeek"] = df.groupby("Company")["SalesScaled"].rolling(7).mean()

df.dropna(inplace=True)

# 1) split on traintest / validation
validation_size = 7 * 4 * 3 # 3 companies, 1 month (7 days times 4)
validation_df = df.tail(validation_size)
traintest_df = df.head(-validation_size)

# 2) train and test and tune hyperparamaters on traintest set

splits = 5
tscv = TimeSeriesSplit(n_splits = splits, test_size = validation_size)


rfc = RandomForestRegressor(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    # 'max_features': [1, 'sqrt', 'log2'],
    'max_features':np.arange(0.1, 1, 0.1),
    'max_depth' : [3, 5, 7, 9],
    # 'criterion' : ["squared_error", "absolute_error"]
}
# param_grid = {'n_estimators':np.arange(50,200,15),
            #   'max_features':np.arange(0.1, 1, 0.1),
            #   'max_depth': [3, 5, 7, 9],
            #   'max_samples': [0.3, 0.5, 0.8]}
# param_grid = {'max_depth' : [3, 5]}
# CV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = tscv, verbose = 2)
# CV_rfc.fit(traintest_df.drop(["SalesScaled", "Date"], axis = 1), traintest_df["SalesScaled"])


# print(CV_rfc.best_params_)


# 3) use found hyperparamaters to train on entire traintest set


# run 1 gives hyperparamters: {'max_depth': 8, 'max_features': 'sqrt', 'n_estimators': 500}
rf_param = RandomForestRegressor(max_depth = 9, max_features =  0.30000000000000004, n_estimators =  500)
rf_param.fit(traintest_df.drop(["SalesScaled", "Date"], axis = 1), traintest_df["SalesScaled"])

# 4) get result on validation set

y_pred = rf_param.predict(validation_df.drop(["SalesScaled", "Date"], axis = 1))

prediction_df = pd.DataFrame()

prediction_df.insert(0, "Company", validation_df["Company"].values)
prediction_df.insert(1, "PredictionScaled", y_pred)

print(prediction_df.head(5))

validation_df.insert(len(validation_df.columns), "PredictionScaled", y_pred)

# validation_df.insert(len(validation_df.columns), "Min", 0.0)
# validation_df.insert(len(validation_df.columns), "Max", 0.0)

# validation_df.loc[validation_df["Company"] == 0, "Min"] = 17.522332908595782
# validation_df.loc[validation_df["Company"] == 0, "Max"] = 164477.3347483395
# validation_df.loc[validation_df["Company"] == 1, "Min"] = -45.04917947145333
# validation_df.loc[validation_df["Company"] == 1, "Max"] = 240464.1980553548
# validation_df.loc[validation_df["Company"] == 2, "Min"] = 9958.738964463202
# validation_df.loc[validation_df["Company"] == 2, "Max"] = 1659719.698434384

# Old Albin solution
# c_min_max = {
#     0: {"Min": 17.522332908595782, "Max":164477.3347483395},
#     1: {"Min": -45.04917947145333, "Max": 240464.1980553548},
#     2: {"Min": 9958.738964463202, "Max": 1659719.698434384}
# }
# validation_df["Min"] = validation_df["Company"].map(lambda x: c_min_max[x]["Min"])
# validation_df["Max"] = validation_df["Company"].map(lambda x: c_min_max[x]["Max"])
# validation_df["Sales"] = validation_df["Company"].map(lambda x: "PredictionScaled" * (c_min_max[x]["Max"] - c_min_max[x]["Min"]) + c_min_max[x]["Min"])

##

# validation_df[["Prediction", "Sales"]] = validation_df.apply(lambda row: calculate_prediction_sales(row))

# def calculate_prediction_sales(row):
#     c_min_max = {
#         0: {"Min": 17.522332908595782, "Max":164477.3347483395},
#         1: {"Min": -45.04917947145333, "Max": 240464.1980553548},
#         2: {"Min": 9958.738964463202, "Max": 1659719.698434384}
#     }
#     min_value = c_min_max[row["Company"]]["Min"]
#     max_value = c_min_max[row["Company"]]["Max"]
#     predict_scaled = row["PredictionScaled"]
#     sales_scaled = row["SalesScaled"]

#     prediction = predict_scaled * (max_value - min_value) + min_value
#     sales = sales_scaled * (max_value - min_value) + min_value
#     return prediction, sales    

# ##

validation_df["Prediction"] = validation_df["PredictionScaled"] * (validation_df["Max"] - validation_df["Min"]) + validation_df["Min"]
validation_df["Sales"] = validation_df["SalesScaled"] * (validation_df["Max"] - validation_df["Min"]) + validation_df["Min"]
print(validation_df["Prediction"].values)


# 4.1) Get error metrics
# RMSE, MAE, WMAPE
# Put these in table in report for the model
rmse = mean_squared_error(validation_df["Prediction"], validation_df["Sales"], squared = False)
wmape = weighted_mean_absolute_percentage_error(validation_df["Prediction"], validation_df["Sales"])
mape = mean_absolute_percentage_error(validation_df["Prediction"], validation_df["Sales"])
mae = mean_absolute_error(validation_df["Prediction"], validation_df["Sales"])

print(f"RMSE: {rmse:.4f}")
print(f"WMAPE: {wmape:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"MAE: {mae:.4f}")

# 5) train on entire dataset and predict on future set