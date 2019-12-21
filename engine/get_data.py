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

from settings import API_KEY

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

def get_CDP_data():
	'''
	Retrieve data on CDPs from Maker API.
	'''
	query = """
	{
	allCups(
		condition: { deleted: false },
		orderBy: RATIO_ASC
	) {
		totalCount
		pageInfo {
		hasNextPage
		hasPreviousPage
		endCursor
		}
		nodes {
		id
		pip
		ire
		tab
		lad
		art
		ink
		ratio
		actions(first: 1) {
			nodes {
			act
			time
			}
		}
		}
	}
	}
	"""
	request = requests.post('https://sai-mainnet.makerfoundation.com/v1', json={'query': query})

	if request.status_code == 200:
		return request.json()
	else:
		raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


#Make into DataFrame
def create_df(from_sym: str, to_sym: str, allData: str = 'true'):
	#Retrieve data from the API
	api_response = get_cryptocompare_data(from_sym = from_sym, to_sym = to_sym, allData = allData).json()
	df = pd.io.json.json_normalize(api_response['Data']['Data'])
	df['time'] = pd.to_datetime(df['time'], unit = 's')
	df = df.set_index('time')
	df.replace(0.0, np.nan, inplace=True)
	return df

def create_close_df():
	#Build matrix of close prices for MCD tokens
	close_master = pd.DataFrame()
	for token in constants.MCD_TOKENS:
		token_df = create_df(from_sym = token, to_sym = 'USDT')
		close = token_df['close']
		close = close.rename(token)
		close_master = pd.concat([close_master, close], axis = 1)
		close_master.replace(0.0, np.nan, inplace=True)
	return close_master

def create_logrets_df():
	#Build matrix of close prices for MCD tokens
	logreturns_master = pd.DataFrame()
	for token in constants.MCD_TOKENS:
		token_df = create_df(from_sym = token, to_sym = 'USD')
		token_df.replace(0.0, np.nan, inplace=True)
		rets = np.log(token_df['close']) - np.log(token_df['close'].shift(1))
		rets = rets.rename(token)
		logreturns_master = pd.concat([logreturns_master, rets], axis = 1)
	return logreturns_master

def create_logrets_series(from_sym: str, to_sym: str):
	'''
	Create a series of log returns for a given pair.
	'''
	df = create_df(from_sym, to_sym)
	df.replace(0.0, np.nan, inplace=True)
	log_returns = np.log(df['close']) - np.log(df['close'].shift(1))
	log_returns = log_returns.dropna()
	return log_returns