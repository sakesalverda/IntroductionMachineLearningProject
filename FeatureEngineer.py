import numpy as np
import pandas as pd
import datetime as dt
from WeatherData.weatherPrecip import cleanPrecipData
from WeatherData.weatherTemp import cleanTempData
from HolidayData.holiday import holiday# # 


def _salaryDayInMonth(today, delta = 0):
    ### Returns the day in which salary payments take place in that month
    # general salary date in Sweden is the 25th, except
    # if 25th is a saturday, payment is on 24th
    # if 25th is a sunday, payment is on 26th
    
    # Support the option to get the result for the previous month, which is needed
    # when the current day is before the salary payment of that month
    # e.g. on the 12th of Mar, the salary was on the 25th of Feb
    if not (delta == 0 or delta == -1):
        raise Exception("Unsupported value for delta give")
    
    # If the previous month is selected for a today date in January, that is december in the previous year
    deltaYear = 0
    if today.month == 1 and delta == -1:
        deltaYear = -1
        delta = 11
    
    newDate = dt.datetime(today.year+deltaYear, today.month+delta, 25)
    
    if newDate.weekday() == 5:
        newDate = newDate.replace(day=24)
        # return 24
    elif newDate.weekday() == 6:
        newDate = newDate.replace(day=26)
        # return 26
        
    return newDate
    
def daysSinceSalary(today):
    ### Calculate the days since the last salary payment
    salary = _salaryDayInMonth(today)
    
    delta = today - today # small trick to make "delta.days" work if "if" and "elif" statement are both not executed
    
    if today.day > salary.day:
        delta = today - salary
    elif today.day < salary.day:
        prevSalary = _salaryDayInMonth(today, delta=-1)
        delta = today - prevSalary
        
    return delta.days


def determineHoliday(dt, holidayData):
    important_list = ["New Year Day", "New Year Eve", "Christmas Eve", "Midsummer Day"]

    holidayLabel = holidayData[(holidayData["Year"] == dt.year) & (holidayData["Month"] == dt.month) & (holidayData["Day"] == dt.day)]["Holiday"]

    if len(holidayLabel.values) > 0:
        holidayLabel = holidayLabel.values[0]
        
    else:
        holidayLabel = None

    if holidayLabel in important_list:
        return 2
    elif holidayLabel: # non-None value
        return 1
    
    return 0

def getTemperature(dt, temperatureData):
    return 20

def getPrecipitation(dt, precipitationData):
    return 0



def processDate(df):
    # Convert into basic year, month, day and weekday
    df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d")
    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    # dayColumns = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # currentDayIndex = df["Date"].dt.isoweekday()
    # currentDayColumn = dayColumns[currentDayIndex]

    # df[currentDayColumn] = 1
    # df[dayColumns[~currentDayColumn]] = 0
    
    df["Monday"] = df["Weekday"] == 0
    df["Tuesday"] = df["Weekday"] == 1
    df["Wednesday"] = df["Weekday"] == 2
    df["Thursday"] = df["Weekday"] == 3
    df["Friday"] = df["Weekday"] == 4
    df["Saturday"] = df["Weekday"] == 5
    df["Sunday"] = df["Weekday"] == 6
    
    # Calculate days since last salary
    df["DaysSinceSalary"] = df["Date"].apply(lambda dt: daysSinceSalary(dt))

    #fdsfgds
    holidayData = holiday()
    df["Holiday"] = df["Date"].apply(lambda dt: determineHoliday(dt, holidayData))
    
    # Determine season
    # df["Season"] = df["Date"].apply(lambda dt: season(dt))
    
    # Include weather data
    tempData = cleanTempData()
    df["Temperature"] = df["Date"].apply(lambda dt: getTemperature(dt, tempData))

    # precipData = cleanPrecipData()
    # df["Precipitation"] = df["Date"].apply(lambda dt: getPrecipitation(dt, prepData))
    
    # Drop date column as the model should not use it
    # df = df.drop(["Date"], axis = 1)
    
    # Sort the dataset, by date and by company
    df.sort_values(by = ["Date", "Company"], inplace = True)

    df = df.drop(["Weekday"], axis = 1)
    
    return df


def season(today):
    # No clear implementation for season in Sweden yet
    return 

def addMissingDates(df, date_range):
    ### Add NaN sales for dates that do not exist in current range
    ### These values will later be filled
    
    # Loop over the three companies
    for i in range(3):
        # Find dates for which no values are given
        missingDates = date_range.difference(df[df["Company"] == i]["Date"])
        
        # Create a new dataframe, set the dates, sales and respective company
        missingDatesDf = pd.DataFrame({"Date": missingDates})
        missingDatesDf["Sales"] = None
        missingDatesDf["Company"] = i
        
        # Add the new data to the main dataframe
        df = pd.concat([df, missingDatesDf])
    
    return df



# The date range of the data
salesRange = pd.date_range(start = "2020-01-01", end = "2023-01-04")

# File names
historical_set = "caspecoHistoricalData.csv"
predict_set = "caspecoTestRange.csv"
processed_set = "caspecoHistoricalDataProcessed.csv" # this file will include all the processed features



hist_df = pd.read_csv(historical_set)
# test_df = pd.read_csv(test_set)

hist_df = addMissingDates(hist_df, date_range = salesRange) # add missing dates

hist_df = processDate(hist_df) # feature engineer with respect to date
# test_df = processDate(test_df)

hist_x = hist_df.loc[:, hist_df.columns != "Sales"]
hist_y = hist_df.loc[:, hist_df.columns == "Sales"]

hist_df.to_csv(processed_set, index=False)

# Just to get an overview of what the data currently looks like
print(hist_x.head())
#print(hist_y.head())


# missingSales = hist_x[hist_y["Sales"].isnull()]

# print(f"There are missing values for {len(missingSales['Date'].unique())} dates\n")
# # Missing values for 
# for i in range(2):
#     print(f"For company {i} the following dates are missing")
#     print(missingSales[missingSales["Company"] == i]["Date"])