import pandas as pd
import numpy as np

from engine import get_data
import constants

def make_close_matrix():
	#Build matrix of close prices for MCD tokens
	close_master = pd.DataFrame()
	for token in constants.MCD_TOKENS:
		token_df = get_data.create_df(from_sym = token, to_sym = 'USDT')
		close = token_df['close']
		close = close.rename(token)
		close_master = pd.concat([close_master, close], axis = 1)
		close_master.replace(0.0, np.nan, inplace=True)
	return close_master

def make_logreturns_matrix():
	#Build matrix of close prices for MCD tokens
	logreturns_master = pd.DataFrame()
	for token in constants.MCD_TOKENS:
		token_df = get_data.create_df(from_sym = token, to_sym = 'USDT')
		token_df.replace(0.0, np.nan, inplace=True)
		rets = np.log(token_df['close']) - np.log(token_df['close'].shift(1))
		rets = rets.rename(token)
		logreturns_master = pd.concat([logreturns_master, rets], axis = 1)
	return logreturns_master