import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import time

time_now = int(time.time())

def fetch_data():
	'''
	Fetch Grin/USDC price data from the Poloniex exchange, for the period 01 Nov until 21 Nov. 
	'''
	POLONIEX_PUBLIC_ENDPOINT = 'https://poloniex.com/public'
	query_params = {'command':'returnChartData', 'currencyPair': 'USDT_ETH', 'period': '1800', 'start': '1546300800', 'end': str(time_now)}
	data = requests.get(url = POLONIEX_PUBLIC_ENDPOINT, params = query_params)
	data = data.json()
	df = pd.DataFrame(data)
	df['log_returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
	df['timestamp'] = df['date']
	df['date'] = pd.to_datetime(df['date'], unit='s')
	df = df.set_index('date')
	return df

def fetch_orderbook():
	'''
	Fetch GRIN/USDC orderbook.
	'''
	POLONIEX_PUBLIC_ENDPOINT = 'https://poloniex.com/public'

	query_params = {'command':'returnTradeHistory', 'currencyPair': 'USDC_GRIN', 'depth': '1000', 'start': '1546300800', 'end': str(time_now)}

	data = requests.get(url = POLONIEX_PUBLIC_ENDPOINT, params = query_params)

	data = data.json()

	return data