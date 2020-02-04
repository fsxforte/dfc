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

from settings import API_KEY
from constants import TOKEN_BASKET
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
	df.replace(0.0, np.nan, inplace=True)
	return df

def create_close_df():
	#Build matrix of close prices for MCD tokens
	close_master = pd.DataFrame()
	for token in constants.TOKEN_BASKET:
		token_df = create_df(from_sym = token, to_sym = 'USD')
		close = token_df['close']
		close = close.rename(token)
		close_master = pd.concat([close_master, close], axis = 1)
		close_master.replace(0.0, np.nan, inplace=True)
	return close_master

def create_logrets_df():
	#Build matrix of close prices for MCD tokens
	logreturns_master = pd.DataFrame()
	for token in constants.TOKEN_BASKET:
		token_df = create_df(from_sym = token, to_sym = 'USD')
		token_df.replace(0.0, np.nan, inplace=True)
		rets = np.log(token_df['close']) - np.log(token_df['close'].shift(1))
		rets = rets.rename(token)
		logreturns_master = pd.concat([logreturns_master, rets], axis = 1)
	return logreturns_master

def create_logrets_series(df):
	'''
	Create a series of log returns on close prices for a given pair.
	:df: input DataFrame, OHLCV format
	'''
	df.replace(0.0, np.nan, inplace=True)
	log_returns = np.log(df['close']) - np.log(df['close'].shift(1))
	log_returns = log_returns.dropna()
	return log_returns

def get_hourly_cryptocompare_data(from_sym: str, to_sym: str, start_date: dt.datetime, end_date: dt.datetime, exchange: str = None):
	'''
	Function to retrieve hourly data from cryptocompare. 	
	
	:tryConversion:	If set to false, it will try to get only direct trading values [ Default - true]
	:fsym:	REQUIRED The cryptocurrency symbol of interest [ Min length - 1] [ Max length - 10]
	:tsym:	REQUIRED The currency symbol to convert into [ Min length - 1] [ Max length - 10]
	:e:	The exchange to obtain data from [ Min length - 2] [ Max length - 30] [ Default - CCCAGG]
	:aggregate:	Time period to aggregate the data over (for daily it's days, for hourly it's hours and for minute histo it's minutes) [ Min - 1] [ Max - 30] [ Default - 1]
	:aggregatePredictableTimePeriods:	Only used when the aggregate param is also in use. If false it will aggregate based on the current time.If the param is false and the time you make the call is 1pm - 2pm, with aggregate 2, it will create the time slots: ... 9am, 11am, 1pm.If the param is false and the time you make the call is 2pm - 3pm, with aggregate 2, it will create the time slots: ... 10am, 12am, 2pm.If the param is true (default) and the time you make the call is 1pm - 2pm, with aggregate 2, it will create the time slots: ... 8am, 10am, 12pm.If the param is true (default) and the time you make the call is 2pm - 3pm, with aggregate 2, it will create the time slots: ... 10am, 12am, 2pm. [ Default - true]
	limit:	The number of data points to return. If limit * aggregate > 2000 we reduce the limit param on our side. So a limit of 1000 and an aggerate of 4 would only return 2000 (max points) / 4 (aggregation size) = 500 total points + current one so 501. [ Min - 1] [ Max - 2000] [ Default - 168]
	:toTs:	Returns historical data before that timestamp. If you want to get all the available historical data, you can use limit=2000 and keep going back in time using the toTs param. You can then keep requesting batches using: &limit=2000&toTs={the earliest timestamp received}
	:explainPath:	If set to true, each point calculated will return the available options it used to make the calculation path decision. This is intended for calculation verification purposes, please note that manually recalculating the returned data point values from this data may not match exactly, this is due to levels of caching in some circumstances. [ Default - false]
	:extraParams:	The name of your application (we recommend you send it) [ Min length - 1] [ Max length - 2000] [ Default - NotAvailable]
	:sign:	If set to true, the server will sign the requests (by default we don't sign them), this is useful for usage in smart contracts [ Default - false]
	'''

	api_endpoint = 'https://min-api.cryptocompare.com/data/v2/histohour'

	now_date = dt.datetime.now()

	delta = now_date - start_date
	total_hours = delta.days*24

	num_loops = int(total_hours/2000 + 1)

	tots_list = []
	tots_list.append(int(mktime(dt.datetime.now().timetuple())))
	for i in range(1, num_loops+1):
		date_time_tot = now_date-timedelta(hours = (i * 2000)) 	
		unix_secs = int(mktime(date_time_tot.timetuple()))
		tots_list.append(unix_secs)

	df_master = pd.DataFrame()
	for i in tots_list:
		params = {
			'fsym': from_sym,
			'tsym': to_sym,
			#'e': exchange,
			'limit': 2000,
			'toTs': i,
		}
		response = requests.get(url=api_endpoint, params=params, headers={'authorization': API_KEY})
		df = pd.DataFrame.from_dict(response.json()['Data']['Data'])
		df_master = pd.concat([df[:-1], df_master]).reset_index(drop = True)
		
	df_master['time'] = pd.to_datetime(df_master['time'], unit = 's')
	df_master = df_master.set_index('time')
	
	return df_master[start_date:end_date]

def get_crisis_endpoints(start_date: dt.datetime, end_date: dt.datetime):
    '''
    Extract the exact dates of the ETH price at the start of 2018.
	:start_date: start date of window within which to search for min/max
	:end_date: end date of window within which to search for min/max
    '''
    #Get ETH price for data window data
    df_eth = create_close_df()['ETH'][start_date:end_date]

    #Extract period of ETH price crash
    start_date_crash = df_eth.idxmax()
    end_date_crash = df_eth.idxmin()
    return start_date_crash, end_date_crash

def liquidity_on_date(token: str, start_date_data: dt.datetime, end_date_data: dt.datetime, date_of_interest: dt.datetime):
    '''
    Extract the volume of a particular token on a particular date.
    '''
    df = create_df(token, 'USD')[start_date_data:end_date_data]
    vol = df.loc[date_of_interest]['volumefrom']
    return vol

def extract_index_of_worst_eth_sim(price_simulations):
	'''
	Find which of the simulator runs resulted in the lowest ETH price at the end. 
	:price_simulations: data from the Monte Carlo simulations. 
	'''
	index = TOKEN_BASKET.index('ETH')
	sims_eth = simulation.asset_extractor_from_sims(price_simulations, index)
	df_eth = pd.DataFrame(sims_eth)
	return df_eth.iloc[-1].nsmallest(1).index

def get_coingecko_data(symbol: str):
	'''
	Retrieve coin data from CoinGecko.
	See https://www.coingecko.com/api/documentations/v3#/coins/get_coins__id__market_chart_range
	for more information.
	:symbol: base symbol, e.g. 'dai'
	'''
	api_endpoint = 'https://api.coingecko.com/api/v3/coins/' + symbol

	return requests.get(url=api_endpoint)

def coingecko_volumes(token: str, volume_unit: str = None):
	'''
	Take raw output from coingecko and extract volume information for each token, returning total 24 hour volume. 
	:coingecko_data_payload: coingecko data payload returned from API.
	:token: token of interest, e.g. 'USDC', 'BAT', 'ETH'
	:volume_unit: unit for the volumes, default is native token but can be returned in DAI by passing 'DAI'
	'''

	coingecko_data_payload = get_coingecko_data('dai')
	tickers = coingecko_data_payload.json()['tickers']
	df = pd.DataFrame(tickers)

	df = df[df['is_stale'] == False]

	##Where DAI is base
	df_dai_base = df[(df['base'] == 'DAI') & (df['target'] == token)]

	#Volume in DAI
	df_dai_base_vol_dai = df_dai_base['volume'].sum()

	#Volume in token
	df_dai_base_vol_token_helper = df_dai_base['volume']*df_dai_base['last']
	df_dai_base_vol_token = df_dai_base_vol_token_helper.sum()

	##Where DAI is target
	df_dai_target = df[(df['target'] == 'DAI') & (df['base'] == token)]

	#Volume in DAI
	df_dai_target_vol_dai_helper = df_dai_target['volume'] * df_dai_target['last']
	df_dai_target_vol_dai = df_dai_target_vol_dai_helper.sum()

	#Volume in token
	df_dai_target_vol_token = df_dai_target['volume'].sum()

	##Overall totals
	dai_vols = df_dai_base_vol_dai + df_dai_target_vol_dai
	token_vols = df_dai_base_vol_token + df_dai_target_vol_token

	if volume_unit == 'DAI':
		return dai_vols
	else:
		return token_vols

def get_liquidities(token_basket):
	'''
	Extract liquidities for all tokens in basket.
	'''
	liquidity_dict = {}
	for token in token_basket:
		liquidity_dict[token] = coingecko_volumes(token = token)
	return liquidity_dict



#Remember to change volumes to liquities!!!! From coingecko
