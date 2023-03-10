# 0) Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Helpers.metrics import error_metrics
from Helpers.df import split_df, add_prediction_to_df, get_train_vars_df, one_hot_encode_df
from Helpers.model import tunecv, forecast, plot, add_lag_features

from sklearn.neural_network import MLPRegressor as Regressor

identifier = "MLP"
linearModel = False


# 1) Read the dataframe
df = pd.read_csv("caspecoHistoricalDataProcessed.csv")


# 2) Add features, that are not sensitive to data leakage

if linearModel == True:
    df = one_hot_encode_df(df)

add_lag_features(df)

df.dropna(inplace=True)



# 3) Split the dataframe into traintest and validation
train_df, validation_df = split_df(df)



# 4) Optimise hyperparamaters
# param_grid = param_list = {"hidden_layer_sizes": [(1,),(50,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0005]}
# param_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
#           'activation': ['relu','tanh','logistic'],
#           'alpha': [0.0001, 0.05, 0.1],
#           'learning_rate': ['constant','adaptive'],
#           'solver': ['adam']}

# rfr = Regressor()
# cv_rfr = tunecv(get_train_vars_df(train_df), train_df["SalesScaled"], rfr, param_grid = param_grid)

# print(cv_rfr.best_params_)
# output: {'activation': 'identity', 'alpha': 0.0005, 'hidden_layer_sizes': (50,), 'solver': 'lbfgs'}
# output2: {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
# output3 (with lags): {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}

# exit() # use during hyper paramater testing

# 5) Train on entire train_df using optimised hyper paramaters

best_params_grid = {'activation': 'identity', 'alpha': 0.05, 'hidden_layer_sizes': (50,), 'solver': 'lbfgs'}
best_params_grid = {"activation": 'identity',
                       "solver": 'adam',"max_iter": 1000}
# best_params_grid = {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
# best_params_grid = {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
model = Regressor(**best_params_grid, random_state = 8)
model.fit(get_train_vars_df(train_df), train_df["SalesScaled"])



# 6) Predict on validation set and reverse the minmax scaling
y_pred = model.predict(get_train_vars_df(validation_df))

add_prediction_to_df(validation_df, y_pred)



# 7) Calculate error metrics
metrics = error_metrics(validation_df)

print(metrics)



# 8) plot against actual, and perhaps baseline
plot(validation_df, name = identifier)


# 9) Use model to predict future set
model = Regressor(**best_params_grid, random_state = 8)
model.fit(get_train_vars_df(df), df["SalesScaled"])
forecast(model, name = identifier, one_hot = linearModel)