import yfinance as yf #pip install yfinance
import pandas as pd
from pandas import read_csv
import datetime as dt
from datetime import timedelta, datetime, date
from dateutil import parser
import csv
from time import sleep
from google.cloud import storage #pip install google-cloud-storage, pip install --upgrade google-cloud-storage, pip install --upgrade google-api-python-client

def data(request):
  window = 250
  storage_client = storage.Client()
  bucket = storage_client.bucket('raw-data_bucket')
  
  blob = bucket.blob('predict_poc.csv')
  with blob.open("r") as f:
    upside = pd.read_csv(f)
  ticker_list = upside['ticker']
  
  blob = bucket.blob('input_poc.csv')
  with blob.open("r") as f:
    data = pd.read_csv(f)
  max_date = data['date'].max()
  
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
        temp = yf.download(ticker)[['Open',	'High',	'Low', 'Close', 'Adj Close', 'Volume']].reset_index().sort_values(by=['Date']).rename(columns={'Date': 'date'})
        if len(temp) == 0:
          sleep(1)
        else:
            error = 0

      temp['real'] = temp[::-1]['Adj Close'].rolling(window).max()[::-1]

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

      ticker_data['ticker'] = ticker
      data_temp_list.append(ticker_data)
      sleep(0) # limit api calls 300 requests 10 seg

  for index in range(0,len(data_temp_list)):
      temp = data_temp_list[index].rename(columns={interval: 'date'})
      if index == 0:
        data_temp = temp
      else:
        data_temp = data_temp.append(temp)
  
  data_temp['date'] = pd.to_datetime(data_temp['date'])

  try:
    data_temp = data_temp[data_temp.date > max_date]
  except:
    print('first load')

  data_temp = data_temp[data_temp.date > max_date]
  data_temp = data_temp.rename(columns={'Dividends': 'dividends'})
  data_temp = data_temp.rename(columns={'Stock Splits': 'splits'})
  data_temp = data_temp.rename(columns={'Open': 'open'})
  data_temp = data_temp.rename(columns={'High': 'high'})
  data_temp = data_temp.rename(columns={'Low': 'low'})
  data_temp = data_temp.rename(columns={'Close': 'close'})
  data_temp = data_temp.rename(columns={'Adj Close': 'adj_close'})
  data_temp = data_temp.rename(columns={'Volume': 'volume'})
  
  blob = bucket.blob('input_poc.csv')
  with blob.open("w") as f:
    data_temp.to_csv(f, index=False)
  
  return f'Success!'
