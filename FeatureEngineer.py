import numpy as np
import pandas as pd
import datetime as dt
from WeatherData.weatherPrecip import cleanPrecipData
from WeatherData.weatherTemp import cleanTempData
from HolidayData.holiday import holiday


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


def getDataForDay(dt, df): 
    return df[(df["Year"] == dt.year) & (df["Month"] == dt.month) & (df["Day"] == dt.day)]

def determineHoliday(dt, holidayData):
    important_list = ["New Year Day", "New Year Eve", "Christmas Eve", "Midsummer Day"]

    holidayLabel = getDataForDay(dt, holidayData)["Holiday"]

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
    dayTemperature = getDataForDay(dt, temperatureData)["Lufttemperatur"]

    return dayTemperature.values[0]

def getPrecipitation(dt, precipitationData):
    dayPrecipitation = getDataForDay(dt, precipitationData)["Nederbördsmängd"]

    return dayPrecipitation.values[0]



def processFeatures(df, advancedFeatures = True, fillMissingSales = True):
    # Convert into basic year, month, day and weekday
    df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d")
    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    # dayColumns = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # for (idx, dayColumn) in enumerate(dayColumns):
    #     df[dayColumn] = df["Weekday"] == idx
    # 
    # df[dayColumns] = df[dayColumns].replace({True: 1, False: 0})
    
    # Calculate days since last salary
    df["DaysSinceSalary"] = df["Date"].apply(lambda dt: daysSinceSalary(dt))

    df["Season"] = df["Date"].apply(lambda dt: seasonForDate(dt))

    if fillMissingSales == True:
        if "Sales" in df.columns:
            # This will get the mean for that company on that weekday
            # @todo
            # Maybe we should do only (previous current and next month) to make it more go with the seasons
            # or perhaps only X last weeks and Y next weeks?
            #
            # I found to categorise by weekday, company (and possiblt season) to be the most accurate
            # I tried to include the year as well, but this does not work for company 0 and 2023 for both comp. 0 and 1
            # (for company 0 several values are still missing if year is included, partly due to it being closed every sunday in the winter of 2022
            # which means no estimations can be made)

            # both of these implementations have the same results but one might be easier to use to get X/Y last/next weeks
            # df['Sales'] = df['Sales'].fillna(df.groupby(['Weekday', 'Company', 'Season'])['Sales'].transform('mean'))
            df['Sales'] = df.groupby(['Weekday', 'Company', 'Season'])['Sales'].transform(lambda x: x.fillna(x.mean()))

    # For computation, sometimes the holiday and weather data is not required
    if advancedFeatures == True:
        # Include holiday data
        holidayData = holiday()
        df["Holiday"] = df["Date"].apply(lambda dt: determineHoliday(dt, holidayData))

        # df["NoHoliday"] = (df["Holiday"] == 0).astype("int")
        # df["ImHoliday"] = (df["Holiday"] == 2).astype("int")
        # df["Holiday"] = (df["Holiday"] == 1).astype("int")
        
        # Include weather data
        tempData = cleanTempData()
        df["Temperature"] = df["Date"].apply(lambda dt: getTemperature(dt, tempData))

        # Include precipitation data
        precipData = cleanPrecipData()
        df["Precipitation"] = df["Date"].apply(lambda dt: getPrecipitation(dt, precipData))
    
    # Drop date column as the model should not use it
    # df = df.drop(["Date"], axis = 1)
    
    # Sort the dataset, by date and by company
    df.sort_values(by = ["Date", "Company"], inplace = True)

    # df["SalesNextDay"] = df.groupby("Company")["Sales"].shift(-1)

    # Here we can add lag, we DON'T implement that here due to data leakage
    # df = addLag(df)

    # @Todo, should we also drop "Day"? We now have daysSinceSalary, which day of the week and whether it is a holiday
    # But it can be usefull to identify the date of it if we run into trouble somewhere...
    # df = df.drop(["Weekday", "Date", "Day", "Season"], axis = 1)
    # df = df.drop(["Weekday", "Date"], axis = 1)
    df = df.drop(["Date"], axis = 1)
    
    return df



def seasonForDate(today):
    # spring = 0
    # summer = 1
    # fall = 2
    # winter = 3

    # using seasonal dates from
    # https://www.calendardate.com/year2023.php

    # No clear implementation for season in Sweden yet
    if today.year == 2020:
        if today.month < 3 or (today.month == 3 and today.day < 19):
            return 3
        elif today.month < 6 or (today.month == 6 and today.day < 20):
            return 0
        elif today.month < 9 or (today.month == 9 and today.day < 22):
            return 1
        elif today.month < 12 or (today.month == 12 and today.day < 21):
            return 2
        else:
            return 3
    elif today.year == 2021:
        if today.month < 3 or (today.month == 3 and today.day < 20):
            return 3
        elif today.month < 6 or (today.month == 6 and today.day < 20):
            return 0
        elif today.month < 9 or (today.month == 9 and today.day < 22):
            return 1
        elif today.month < 12 or (today.month == 12 and today.day < 21):
            return 2
        else:
            return 3
    elif today.year == 2022:
        if today.month < 3 or (today.month == 3 and today.day < 20):
            return 3
        elif today.month < 6 or (today.month == 6 and today.day < 21):
            return 0
        elif today.month < 9 or (today.month == 9 and today.day < 22):
            return 1
        elif today.month < 12 or (today.month == 12 and today.day < 21):
            return 2
        else:
            return 3
    elif today.year == 2023:
        # Data stops before march, thus only winter is possible in 2023
        return 3




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


def getFeaturedData(fillMissingSales = True, generateFiles = True, advancedFeatures = True):
    # The date range of the data
    salesRange = pd.date_range(start = "2020-01-01", end = "2023-01-04")

    # File names
    historical_set = "caspecoHistoricalData.csv"
    predict_set = "caspecoTestRange_sorted.csv"

    processed_historical_set = "caspecoHistoricalDataProcessed.csv" # this file will include all the processed features
    processed_predict_set = "caspecoTestRangeProcessed.csv" # this file will include all the processed features



    hist_df = pd.read_csv(historical_set)
    predict_df = pd.read_csv(predict_set)

    hist_df = addMissingDates(hist_df, date_range = salesRange) # add missing dates

    hist_df = processFeatures(hist_df, advancedFeatures = advancedFeatures, fillMissingSales = fillMissingSales) # feature engineer with respect to date
    predict_df = processFeatures(predict_df, advancedFeatures = advancedFeatures)

    # hist_x = hist_df.loc[:, hist_df.columns != "Sales"]
    # hist_y = hist_df.loc[:, hist_df.columns == "Sales"]

    if generateFiles == True:
        hist_df.to_csv(processed_historical_set, index = False)
        predict_df.to_csv(processed_predict_set, index = False)

    else:
        return hist_df

    missingSales = hist_df[hist_df["Sales"].isnull()]

    # print(f"There are missing values for {len(missingSales['Date'].unique())} dates\n")
    # Missing values for 
    for i in range(3):
        print(f"For company {i} the following dates are missing")
        print(missingSales[missingSales["Company"] == i][["Year", "Month"]])

if __name__ == '__main__':
    getFeaturedData(advancedFeatures = True)