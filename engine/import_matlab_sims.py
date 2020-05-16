from scipy import io
from collections import defaultdict

def get_matlab_sims(sim_version: str):
    '''
    Import a 3d Matlab araay in the format needed for the crash simulator. 
    :sim_version: choose from 'A', 'B' or 'C', depending on which correlation you want
    between the collateral and the reserve asset. 
    '''

    matlab_object = io.loadmat('matlab_data/' + sim_version + '.mat')[sim_version]

    price_simulations = {}

    for day in matlab_object:

        eth_sims_per_day = day[0]
        
        for sim_index, sim in enumerate(eth_sims_per_day):
            if str(sim_index+1) not in price_simulations:
                price_simulations[str(sim_index+1)] = {}
            if 'ETH' not in price_simulations[str(sim_index+1)]:
                price_simulations[str(sim_index+1)]['ETH'] = []

            price_simulations[str(sim_index+1)]['ETH'].append(sim)

        res_sims_per_day = day[1]

        for sim_index, sim in enumerate(res_sims_per_day):
            if 'RES' not in price_simulations[str(sim_index+1)]:
                price_simulations[str(sim_index+1)]['RES'] = []

            price_simulations[str(sim_index+1)]['RES'].append(sim)

    return price_simulations