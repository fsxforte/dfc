import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def fetch_data():
	'''
	Fetch Grin/USDC price data from the Poloniex exchange, for the period 01 Nov until 21 Nov. 
	'''

	POLONIEX_PUBLIC_ENDPOINT = 'https://poloniex.com/public'

	query_params = {'command':'returnChartData', 'currencyPair': 'USDC_GRIN', 'period': '300', 'start': '1572566400', 'end': '1574348631'}

	data = requests.get(url = POLONIEX_PUBLIC_ENDPOINT, params = query_params)

	data = data.json()

	df = pd.DataFrame(data)
	df['log_returns'] = np.log(df['close']) - np.log(df['close'].shift(1))

	return df