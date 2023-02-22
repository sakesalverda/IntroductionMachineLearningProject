import numpy as np
import pandas as pd
import datetime as dt


def _divideDates(df):
    # Divide
    df[["Day", "Month", "Year"]] = df["Datum"].str.split("/", expand = True)
    # Set type to int
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    df.drop(columns="Datum", inplace=True)

def _getTimeperiod(df):
    df.drop(df[df.Hour < 10].index, inplace=True)
    df.drop(df[df.Hour > 20].index, inplace=True)
    
def _timeToInt(df):
    df[["Hour", "Minute", "Second"]] = df["Tid (UTC)"].str.split(":", expand = True) #TODO look to optimize
    df.drop(columns="Minute", inplace=True)
    df.drop(columns="Second", inplace=True)
    df.drop(columns="Tid (UTC)", inplace=True)
    df['Hour'] = df['Hour'].astype(int)

def _meanTemp(df):
    df_mean = df.groupby(["Year","Month","Day"], group_keys = False,as_index = False).mean() 
    df_mean.drop(columns="Hour", inplace=True)
    return df_mean

'''
Concatenate historical and recent Temp, 
convert data to right types (int, float),
get mean of temperature,
dataframe will consist of temp(mean), day, month, year
'''
def cleanTempData():
    # Load csv's
    dfTempHist = pd.read_csv('T_historical.csv', sep = ",", skiprows=10, usecols=[0,1,2]) # TODO wtf why sep = commas
    dfTempRecent = pd.read_csv('T_recent.csv', sep = ",", skiprows=10, usecols=[0,1,2])

    # Remove null data
    dfTempHist=dfTempHist.dropna()
    dfTempRecent=dfTempRecent.dropna()
    
    # Convert time 
    _timeToInt(dfTempRecent)
    _timeToInt(dfTempHist)

    # Divide date by day, month, year
    _divideDates(dfTempRecent)
    _divideDates(dfTempHist)

    # Remove dates before 2020
    dfTempHist.drop(dfTempHist[dfTempHist.Year < 2020].index, inplace=True)

    # Concatenate the two dataframes while dropping duplicates
    dfAll = pd.concat([dfTempHist, dfTempRecent]).drop_duplicates() 

    # Get timespan (10-20)
    _getTimeperiod(dfAll)
 
    # Get mean of each day
    dfAll = _meanTemp(dfAll)

    print(dfAll.describe()) 


cleanTempData()