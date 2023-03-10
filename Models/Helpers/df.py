import pandas as pd
from Helpers.features import calculate_prediction_sales

df_drop_train_columns = ["Date", "SalesScaled"]

# validate on a 4 weeks of data
# validation_size = (4 * 7) * 3 # 3 companies
validation_size = (4 * 7) * 3
test_size = validation_size

def get_rows_for_company_df(df, company):
    if "Company" in df.columns:
        rows = df["Company"] == company
    else:
        rows = df[f"Company_{company}"] == 1
    
    return rows

def one_hot_encode_df(df):
    columns = ["Weekday", "Month", "Season", "Holiday", "Day", "DaysSinceSalary"]

    for column in columns:
        df = pd.get_dummies(df, columns=[column], prefix = column, drop_first=False)
        # one_hot = pd.get_dummies(data = df, prefix = column, columns = [column])

        # print(one_hot)

        # df = df.drop(column, axis = 1)
        # df = df.join(one_hot)

    return df



def split_df(df):
    validation_df = df.tail(validation_size).copy()
    train_df = df.head(-validation_size).copy()

    return train_df, validation_df

def add_prediction_to_df(df, pred):
    df.insert(len(df.columns), "PredictionScaled", pred)
    df.insert(len(df.columns), "Prediction", 0)

    df[["Prediction", "Sales"]] = df.apply(lambda row: calculate_prediction_sales(row), axis = "columns",  result_type='expand')

def get_train_vars_df(df):
    return df.drop(df_drop_train_columns, axis = 1)