import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import time
import os
from urllib.parse import urljoin
import constants
from datetime import timedelta
import datetime as dt
from time import mktime
import requests

from settings import API_KEY
from constants import COLLATERAL_ASSET
from engine import simulation

time_now = int(time.time())

def get_cryptocompare_data(from_sym: str, to_sym: str, exchange: str = None, allData: str = None):
	'''
	Function to retrieve data from cryptocompare. 	
	:fsym:	(REQUIRED) The cryptocurrency symbol of interest [ Min length - 1] [ Max length - 10]
	:tsym:	(REQUIRED) The currency symbol to convert into [ Min length - 1] [ Max length - 10]
	:tryConversion:	If set to false, it will try to get only direct trading values [ Default - true]
	:e:	The exchange to obtain data from [ Min length - 2] [ Max length - 30] [ Default - CCCAGG]
	:aggregate:	Time period to aggregate the data over (for daily it's days, for hourly it's hours and for minute histo it's minutes) [ Min - 1] [ Max - 30] [ Default - 1]
	:aggregatePredictableTimePeriods:	Only used when the aggregate param is also in use. If false it will aggregate based on the current time.If the param is false and the time you make the call is 1pm - 2pm, with aggregate 2, it will create the time slots: ... 9am, 11am, 1pm.If the param is false and the time you make the call is 2pm - 3pm, with aggregate 2, it will create the time slots: ... 10am, 12am, 2pm.If the param is true (default) and the time you make the call is 1pm - 2pm, with aggregate 2, it will create the time slots: ... 8am, 10am, 12pm.If the param is true (default) and the time you make the call is 2pm - 3pm, with aggregate 2, it will create the time slots: ... 10am, 12am, 2pm. [ Default - true]
	:limit:	The number of data points to return. If limit * aggregate > 2000 we reduce the limit param on our side. So a limit of 1000 and an aggerate of 4 would only return 2000 (max points) / 4 (aggregation size) = 500 total points + current one so 501. [ Min - 1] [ Max - 2000] [ Default - 30]
	:allData:	Returns all data (only available on histo day) [ Default - false]
	:toTs:	Returns historical data before that timestamp. If you want to get all the available historical data, you can use limit=2000 and keep going back in time using the toTs param. You can then keep requesting batches using: &limit=2000&toTs={the earliest timestamp received}
	:explainPath:	If set to true, each point calculated will return the available options it used to make the calculation path decision. This is intended for calculation verification purposes, please note that manually recalculating the returned data point values from this data may not match exactly, this is due to levels of caching in some circumstances. [ Default - false]
	:extraParams:	The name of your application (we recommend you send it) [ Min length - 1] [ Max length - 2000] [ Default - NotAvailable]
	:sign:	If set to true, the server will sign the requests (by default we don't sign them), this is useful for usage in smart contracts [ Default - false]
	'''

	api_endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'

	params = {
        'fsym': from_sym,
        'tsym': to_sym,
        'e': exchange,
        'allData': allData,
    }

	return requests.get(url=api_endpoint, params=params, headers={'authorization': API_KEY})

def get_prices(from_sym: str, to_sym: str, start_date: dt.datetime, end_date: dt.datetime, allData: str = 'true', close_only = True):
	'''
	Assemble cryptocompare data into a dataframe.
	:from_sym: base currency
	:to_sym: quote currency
	'''
	#Retrieve data from the API
	api_response = get_cryptocompare_data(from_sym = from_sym, to_sym = to_sym, allData = allData).json()
	df = pd.io.json.json_normalize(api_response['Data']['Data'])
	df['time'] = pd.to_datetime(df['time'], unit = 's')
	df = df.set_index('time')
	df = df[start_date:end_date]

	if close_only:
		df = df['close']

	df = df.rename(str(COLLATERAL_ASSET))

	return df

def compute_log_returns(close_prices):
	'''
	Compute log returns from close prices.
	'''
	log_returns = np.log(close_prices) - np.log(close_prices.shift(1))
	log_returns = log_returns.dropna()

	return log_returns

def extract_index_of_worst_eth_sim(price_simulations, point_evaluate_eth_price: int):
	'''
	Find which of the simulator runs resulted in the lowest ETH price at the end. 
	:price_simulations: data from the Monte Carlo simulations.
	:point_evaluate_eth_price: point at which the ETH price should be evaluated. 
	'''
	sims_eth = simulation.asset_extractor_from_sims(price_simulations, 'ETH')
	df_eth = pd.DataFrame(sims_eth)
	return df_eth.iloc[point_evaluate_eth_price].nsmallest(1).index

def get_defi_pulse_data(length):
	URL = 'https://public.defipulse.com/api/GetHistory'

	params = {'length': 365}

	response = requests.get(url=URL, params=params)

	if response.status_code != 200:
		print('Error retrieving data from the DeFi Pulse API: ' + str(response.status_code))

	df = pd.DataFrame(response.json())

	df['date'] = pd.to_datetime(df['timestamp'], unit='s')
	df_sorted = df.sort_values('date')
	df_sorted = df_sorted.set_index('date')

	return df_sorted

