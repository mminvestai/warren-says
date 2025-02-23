import yfinance as yf #pip install yfinance
import pandas as pd
from pandas import read_csv
import datetime as dt
from datetime import timedelta, datetime, date
from dateutil import parser
import csv
from time import sleep
from google.cloud import storage #pip install google-cloud-storage, pip install --upgrade google-cloud-storage, pip install --upgrade google-api-python-client

def ingest(request):
    storage_client = storage.Client()
    bucket = storage_client.bucket('raw-data_bucket')
    blob = bucket.blob('predict_poc.csv')
    with blob.open("r") as f:
        upside = pd.read_csv(f)

    ticker_list = upside['ticker']
    window = 250 #MMP 25-02-23

    def adj_close(ticker_symbol,start_date="1980-01-01"):
        # Fetch raw stock data
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(start=start_date, auto_adjust=False)
    
        # Get corporate actions
        dividends = stock.dividends
    
        # Step 1: Compute Dividend Factor (cumulative product from future to past)
        df["Dividend Factor"] = 1.0
        cumulative_div = 1.0
        for date in reversed(df.index):
            if date in dividends:
                close_price = df.loc[date, "Close"]
                if close_price > 0:
                    cumulative_div *= (close_price - dividends[date]) / close_price
            df.loc[date, "Dividend Factor"] = cumulative_div  # Apply backwards
    
        # Final Adjusted Close Calculation (Only adjusting for dividends)
        df["Adj Close"] = df["Close"] * df["Dividend Factor"]
    
        return df[['Open',	'High',	'Low', 'Close', 'Adj Close', 'Volume']].reset_index().sort_values(by=['Date']).rename(columns={'Date': 'date'})
    
    def consolidate(df_daily, df_interval, interval):
        if df_interval.empty == True:
            df_interval.rename(columns = {interval: 'date'}, inplace = True)
            df_consolidated = df_daily.merge(df_interval, on = ['date'], how='outer')
        else:
            if interval == 'date':
                df_consolidated = df_daily.merge(df_interval, on = ['date'], how='outer')
            else:
                if interval == 'week':
                    df_interval['week'] = (df_interval['date'].dt.week % 52) + 1
                    df_daily['week'] = df_daily['date'].dt.week
                elif interval == 'month':
                    df_interval['month'] = (df_interval['date'].dt.month % 12) + 1
                    df_daily['month'] = df_daily['date'].dt.month
                elif interval == 'quarter':
                    df_interval['quarter'] = (df_interval['date'].dt.quarter % 4) + 1
                    df_daily['quarter'] = df_daily['date'].dt.quarter
                df_interval['year'] = 0
                df_interval.loc[df_interval[interval] == 1, 'year'] = df_interval['date'].dt.year + 1
                df_interval.loc[df_interval[interval] != 1, 'year'] = df_interval['date'].dt.year
                df_daily['year'] = df_daily['date'].dt.year
                df_interval = df_interval.drop(columns=['date'])
                df_consolidated = df_daily.merge(df_interval, on = [interval, 'year'], how='outer')
                df_consolidated = df_consolidated.drop(columns=['year'])
                df_consolidated = df_consolidated.drop(columns=[interval])
        col = df_consolidated.pop('date')
        df_consolidated.insert(0, 'date', col)
        return df_consolidated

    #def ingest(request):
    data_temp_list = []

    for ticker in ticker_list:
        df_list = []
        # Yahoo Finance Dividend and Splits
        running = True
        while running:
            try:
                temp = yf.Ticker(ticker).history(period="max")[['Dividends', 'Stock Splits']].reset_index().sort_values(by=['Date']).rename(columns={'Date': 'date'})
                running = False
            except:
                sleep(1)
        df_list.append(temp)

        # Yahoo Finance Stock Prices
        error = 1
        while error == 1:
            temp = adj_close(ticker)
            if len(temp) == 0:
                sleep(1)
            else:
                error = 0

        df_list.append(temp)
        
        # Nasdaq Quandl Options Implied Volatility
        #temp = quandl.get("VOL/"+opt_ticker)[['Hv10','Hv180','Phv10','Phv180','IvCall10','IvPut10','IvCall1080','IvPut1080']].reset_index().sort_values(by=['Date']).rename(columns={'Date': 'date'})
        #df_list.append(temp)

        ticker_data = df_list[0]
        #+2022-10-23
        ticker_data['date'] = ticker_data['date'].dt.strftime('%Y-%m-%d')
        temp['date'] = temp['date'].dt.strftime('%Y-%m-%d')
        #-2022-10-23

        if len(df_list) > 0:
            for index in range(1,len(df_list)):
                interval = df_list[index].columns[0]
                temp = df_list[index].rename(columns={interval: 'date'})
                ticker_data = consolidate(ticker_data, temp, interval)

        ticker_data.columns = ['{}{}'.format(c, '' if c == 'date' else '_'+ticker) for c in ticker_data.columns]
        data_temp_list.append(ticker_data)
        sleep(0) # limit api calls 300 requests 10 seg

    for index in range(0,len(data_temp_list)):
        interval = data_temp_list[index].columns[0]
        temp = data_temp_list[index].rename(columns={interval: 'date'})
        #temp['date'] = temp['date'].apply(lambda x: x.strftime('%Y-%m-%d')) -2022-10-23
        if len(data_temp_list[index]) > 0:
            temp['date'] = pd.to_datetime(temp['date'])
        if index == 0:
            data_temp = temp
        else:
            data_temp = consolidate(data_temp, temp, interval)

    data_temp['date_num'] = data_temp['date'].apply(dt.date.toordinal) - 693594 #get it in date_num in excel
    data_temp = data_temp[data_temp.date_num > 0]

    data_temp = data_temp.sort_values(by=['date']).fillna(method="ffill")

    #data_temp.pop('date') MMP 19-02-23

    blob = bucket.blob('data_poc.csv')
    with blob.open("w") as f:
        data_temp.to_csv(f, index=False)
    
    return f'Success!'
