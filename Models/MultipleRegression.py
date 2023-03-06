import numpy as np
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBRegressor

df = pd.read_csv("caspecoHistoricalDataProcessed.csv")
targetDf = pd.read_csv("caspecoTestRangeProcessed.csv")

dfObject = {
    0: df[df["Company"] == 0],
    1:  df[df["Company"] == 1],
    2:  df[df["Company"] == 2]
}

output_df = pd.read_csv("caspecoTestRange_sorted.csv")

for company in range(1):
    dfComp = dfObject[company]
    dfCompPredict = targetDf[targetDf["Company"] == company]

    ### Split data

    # split without timeseries
    x_train = dfComp.drop(["Sales", "Day", "Season"], axis = 1)
    y_train = dfComp["Sales"]

    # split with timeseries
dict_classifiers = {

    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Linear SVM": SVC(),
    #"XGB": XGBRegressor(),
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

params = {
    "Random Forest": {"max_depth": range(5, 30, 5), "min_samples_leaf": range(1, 30, 2),
                      "n_estimators": range(100, 2000, 200)},

    "Gradient Boosting Classifier": {"learning_rate": [0.001, 0.01, 0.1], "n_estimators": range(1000, 3000, 200)},
    "Linear SVM": {"kernel": ["rbf", "poly"], "gamma": ["auto", "scale"], "degree": range(1, 6, 1)},
    "XGB": {'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5], "n_estimators": [300, 600],
            "learning_rate": [0.001, 0.01, 0.1],
            },
    "Logistic Regression": {'penalty': ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    "Nearest Neighbors": {'n_neighbors': [3, 5, 11, 19], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
    "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15)},

}    


splits = 5
custom_cv = []
tscv = TimeSeriesSplit(n_splits = splits, test_size=int(len(x_train) / (splits + 1) - 1))
for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
    custom_cv.append((np.array(train_index.tolist()),np.array(test_index.tolist())))
    for classifier_name in dict_classifiers.keys() & params:

        print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")


        x_train_split, x_test_split = x_train.iloc[train_index, :], x_train.iloc[test_index,:]
        y_train_split, y_test_split = y_train.iloc[train_index], y_train.iloc[test_index]
        print(len(x_train_split))

        #gridSearchCV
        gridSearch = GridSearchCV(estimator=dict_classifiers[classifier_name], param_grid=params[classifier_name], cv=custom_cv)
        gridSearch.fit(x_train_split, y_train_split)

        print(gridSearch.best_score_, gridSearch.best_params_)
        #Linear Regression
        # model = LinearRegression()
        # model.fit(x_train_split, y_train_split)

        # prediction = model.predict(x_test_split)

        # score = model.score(x_test_split, y_test_split)
        # mse = mean_squared_error(y_test_split, prediction)

        # print(f"  Score: {score}")
        # print(f"  MSE: {mse}")




    # x_predict = dfCompPredict.drop(["Day", "Season"], axis = 1)

    # model = LinearRegression()
    # model.fit(x_train, y_train)

    # prediction = model.predict(x_predict)

    # output_df.loc[output_df["Company"] == company, "Sales"] = prediction


    # time_range = np.concatenate((y_train.tail(20).values, prediction))

    # plt.figure()
    # plt.plot(time_range)
    # plt.show()

    # print(prediction)

output_df["ID"] = output_df['Date'] + "_" + output_df['Company'].astype(str)
    

# output_df.to_csv("Predictions/SimpleLinRegPrediction.csv", sep = ",", index = False, columns = ['ID', 'Sales'])
print(output_df.head(10))