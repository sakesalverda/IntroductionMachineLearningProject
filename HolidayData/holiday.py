import pandas as pd



def _divideDates(df):
    # Divide
    df[["Year", "Month", "Day"]] = df["Date"].str.split("-", expand = True)
    # Set type to int
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    # df.drop(columns="Date", inplace=True)

def holiday():
    # read csv
    dfHolidays = pd.read_csv("HolidayData/Holidays.csv", sep = ",", usecols=[0,1])

    # rename "Day" column to "Holiday"
    dfHolidays = dfHolidays.rename(columns={"Day" : "Holiday"})

    # format date
    _divideDates(dfHolidays)
    
    return dfHolidays
