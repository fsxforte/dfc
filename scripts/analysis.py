import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import datetime as dt

from engine import get_data
import constants

def make_correlations_matrix(input_matrix):
	corr_df = input_matrix.corr(method = 'pearson')
	corr_df.head().reset_index()
	del corr_df.index.name
	return corr_df

close_prices = make_close_matrix()
close_prices = close_prices[dt.datetime(2017,12,1):dt.datetime(2018,1,1)]

