import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from engine import get_data
import constants

def make_correlations_matrix(input_matrix):
	corr_df = input_matrix.corr(method = 'pearson')
	corr_df.head().reset_index()
	del corr_df.index.name
	return corr_df

#Get series of liquidation prices for CDPs
data = get_data.get_CDP_data()
df = pd.DataFrame.from_dict(data['data']['allCups']['nodes'])
df.set_index(df['id'], inplace=True)
df['art'] = df['art'].astype(float)
df['ink'] = df['ink'].astype(float)

#Calculate liquidation price
df['liquidation_price'] = df['art'].astype(float) * 1.5 / df['ink'].astype(float)
#df[df['id']==147710]

#Extract subset of liquidation prices
#Subset constraints: (1) more than 5USD in DAI
df = df[df['art']>5]

sns.distplot(df['liquidation_price'])

#Want to see how cumulative eth volume increases with liquidation price
df.sort_values(by = 'liquidation_price', inplace=True, ascending = False)
df['cumulative_eth']=df['ink'].cumsum()

df.plot(x = 'cumulative_eth', y = 'liquidation_price')