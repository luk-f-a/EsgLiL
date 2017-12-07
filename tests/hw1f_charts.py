# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:22:52 2017

@author: CHK4436
"""

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil.esg import ESG
from esglil.common import TimeDependentParameter
from esglil.ir_models import HullWhite1fShortRate, HullWhite1fBondPrice, HullWhite1fCashAccount
from esglil import ir_models

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import datetime
from scipy.stats import normaltest, shapiro, probplot, kstest
import matplotlib.mlab as mlab
import pandas as pd

def plot_short_rate_all_sims():
    delta_t = 1
    n_sims= 10
    dW = rng.NormalRng(dims=1, sims=n_sims, mean=[0], cov=[[delta_t]])
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.01)
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r)
    df_full_run = esg.full_run_to_pandas(dt_out=1, max_t=40)
    ax=None
    initial_r = pd.DataFrame.from_records([(0, i, B(0)) for i in range(n_sims)], 
                                           columns=['time', 'sim', 'r']).set_index(['time', 'sim'])
    df = pd.concat([initial_r, df_full_run[['r']]])
    for sim, r in df.groupby(level='sim'):
        if ax is None:
            ax = r.reset_index('sim')['r'].plot()
        else:
            ax = r.reset_index('sim')['r'].plot(ax=ax)
        

        
def plot_short_rate_pcentiles():

    delta_t = 1/52
    n_sims= 10_000
    dW = rng.NormalRng(dims=1, sims=n_sims, mean=[0], cov=[[delta_t]])
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.01)
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r)
    df_full_run = esg.full_run_to_pandas(dt_out=1, max_t=40)
    initial_r = pd.DataFrame.from_records([(0, i, B(0)) for i in range(n_sims)], 
                                           columns=['time', 'sim', 'r']).set_index(['time', 'sim'])
    
    sims = pd.concat([initial_r, df_full_run[['r']]]).unstack('time').values
   
    
    x = np.arange(41)
    pctiles = [np.percentile(sims, p, axis=0) for p in [0,25,75,100]]
    median = np.mean(sims, axis=0)
    print(median)
    plt.plot(x, median, 'k')
    n = len(pctiles)
    color = plt.cm.Blues(np.linspace(0.4,0.6,n))

    pct_pairs = list(zip(pctiles[:-1], pctiles[1:]))
    for i in range(n//2):
        p1, p2 = pct_pairs[i]
        plt.fill_between(x, p1, p2, facecolor=color[-i], alpha=0.5)
        p1, p2 = pct_pairs[-i-1]
        plt.fill_between(x, p1, p2, facecolor=color[-i], alpha=0.5)
    plt.show()

    
def plot_short_rate_dist():
    now_ = datetime.datetime.now()
    delta_t = 1/5000
    sim_nb = 10000
    dW = rng.NormalRng(dims=1, sims=sim_nb, mean=[0], cov=[[delta_t]])
    T=1
    a = 0.5
    mean_rev_level = 0.05
    sigma = 0.01

    B_fc = lambda t: mean_rev_level + 0.1*t
    B = TimeDependentParameter(function= B_fc)
    b = lambda t: 0.1 #derivative of B_fc
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r)
    df_full_run = esg.full_run_to_pandas(dt_out=1, max_t=T)['r']
    print(datetime.datetime.now()-now_)
    test_time = 1
    sims_test = df_full_run.values
    theoretical_mu = B_fc(T)
    theoretical_sigma = np.sqrt(sigma**2/2/a*(1-np.exp(-2*a*T)))
    n, bins, patches = plt.hist(sims_test, bins=100,  normed=1)
    y = mlab.normpdf(bins, theoretical_mu, theoretical_sigma)
    plt.plot(bins, y, '--')    
    plt.show()
    print('mean', theoretical_mu, sims_test.squeeze().mean())
    print('sigma', theoretical_sigma, sims_test.squeeze().std())


    _, p1 = normaltest(sims_test.squeeze())
#    z = (sims_test.values.squeeze()-theoretical_mu)/theoretical_sigma
#    _, p3= kstest(z, 'norm',args=(0,1))
    
    print('p-value normal test (>0.05 is good)', p1)
    probplot(sims_test.squeeze(),
             sparams=(theoretical_mu,theoretical_sigma),fit=False, plot=plt)
    plt.show()
    
def extract_esg_output():
    now_ = datetime.datetime.now()
    #Based on EUR 17Q2? data
    bond_prices = {1: 1.0073, 2: 1.0111, 3: 1.0136, 4: 1.0138, 5: 1.0115, 6: 1.0072,
          7: 1.0013, 8: 0.9941, 9: 0.9852, 10: 0.9751, 11: 0.9669, 12: 0.9562,
          13: 0.9461, 14: 0.937, 15: 0.9281, 16: 0.9188, 17: 0.9091, 18: 0.8992, 
          19: 0.8894, 20: 0.8797, 21: 0.8704, 22: 0.8614, 23: 0.8527, 24: 0.8442, 
          25: 0.836, 26: 0.828, 27: 0.8202, 28: 0.8127, 29: 0.8053, 30: 0.7983, 
          31: 0.7914, 32: 0.7848, 33: 0.7783, 34: 0.772, 35: 0.7659, 36: 0.7599, 
          37: 0.7539, 38: 0.7481, 39: 0.7423, 40: 0.7366 }
    swaption_prices = [
             {'normal_vol': 0.004216, 'start': 1, 'strike': 'ATM', 'tenor': 1},
             {'normal_vol': 0.00519, 'start': 2, 'strike': 'ATM', 'tenor': 1},
             {'normal_vol': 0.006195, 'start': 3, 'strike': 'ATM', 'tenor': 1},
             {'normal_vol': 0.007097, 'start': 5, 'strike': 'ATM', 'tenor': 1},
             {'normal_vol': 0.007277, 'start': 7, 'strike': 'ATM', 'tenor': 1},
             {'normal_vol': 0.007368, 'start': 10, 'strike': 'ATM', 'tenor': 1},
             {'normal_vol': 0.004332, 'start': 1, 'strike': 'ATM', 'tenor': 2},
             {'normal_vol': 0.00532, 'start': 2, 'strike': 'ATM', 'tenor': 2},
             {'normal_vol': 0.0063, 'start': 3, 'strike': 'ATM', 'tenor': 2},
             {'normal_vol': 0.00729, 'start': 5, 'strike': 'ATM', 'tenor': 2},
             {'normal_vol': 0.007424, 'start': 7, 'strike': 'ATM', 'tenor': 2},
             {'normal_vol': 0.00749, 'start': 10, 'strike': 'ATM', 'tenor': 2},
             {'normal_vol': 0.004378, 'start': 1, 'strike': 'ATM', 'tenor': 3},
             {'normal_vol': 0.005424, 'start': 2, 'strike': 'ATM', 'tenor': 3},
             {'normal_vol': 0.006434, 'start': 3, 'strike': 'ATM', 'tenor': 3},
             {'normal_vol': 0.007391, 'start': 5, 'strike': 'ATM', 'tenor': 3},
             {'normal_vol': 0.007406, 'start': 7, 'strike': 'ATM', 'tenor': 3},
             {'normal_vol': 0.007439, 'start': 10, 'strike': 'ATM', 'tenor': 3},
             {'normal_vol': 0.004571, 'start': 1, 'strike': 'ATM', 'tenor': 5},
             {'normal_vol': 0.005691, 'start': 2, 'strike': 'ATM', 'tenor': 5},
             {'normal_vol': 0.006551, 'start': 3, 'strike': 'ATM', 'tenor': 5},
             {'normal_vol': 0.007506, 'start': 5, 'strike': 'ATM', 'tenor': 5},
             {'normal_vol': 0.007473, 'start': 7, 'strike': 'ATM', 'tenor': 5},
             {'normal_vol': 0.007376, 'start': 10, 'strike': 'ATM', 'tenor': 5},
             {'normal_vol': 0.004941, 'start': 1, 'strike': 'ATM', 'tenor': 7},
             {'normal_vol': 0.00593, 'start': 2, 'strike': 'ATM', 'tenor': 7},
             {'normal_vol': 0.006794, 'start': 3, 'strike': 'ATM', 'tenor': 7},
             {'normal_vol': 0.0076, 'start': 5, 'strike': 'ATM', 'tenor': 7},
             {'normal_vol': 0.007646, 'start': 7, 'strike': 'ATM', 'tenor': 7},
             {'normal_vol': 0.007412, 'start': 10, 'strike': 'ATM', 'tenor': 7}]
    a = 0.01
    sigma = ir_models.hw1f_sigma_calibration(bond_prices, 
                                                swaption_prices, a)

    B =  ir_models.hw1f_B_function(bond_prices, a, sigma)
    delta_t = 1/250
    dW = rng.NormalRng(dims=1, sims=1000, mean=[0], cov=[[delta_t]])
    B = TimeDependentParameter(function=B)
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    #T = np.array(list(bond_prices.keys()))
    P = {'Bond_{}'.format(i):HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
                             P_0=bond_prices[i], T=i) for i in range(1,41)}
    C = HullWhite1fCashAccount(r=r)
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r, cash=C, **P)

    df_sims = esg.full_run_to_pandas(dt_out=1, max_t=40)
    print('time to simulate: ',datetime.datetime.now()-now_)
    #Calculate rate and arrange them by time to maturity (instead of maturity as the bonds)
    df_sims = df_sims.swaplevel(0,1)
    df_sims = df_sims.stack()
    df_sims.name = 'value'
    df_sims.index.names = ['sim','time', 'var']
    df_sims = df_sims.reset_index('var')
    df_bonds = df_sims[df_sims['var'].str.startswith('Bond_')]
    
    df_cash = df_sims[df_sims['var'].str.startswith('cash')]
    mat = df_bonds['var'].str.split('_').str.get(1).astype('float').astype('int')
    t = df_bonds.index.get_level_values('time').astype('int')
    df_bonds.loc[:,'time2mat'] = (mat - t)
    df_rates = df_bonds.copy()
    df_rates.loc[:,'var'] = 'rate_'+df_bonds['time2mat'].astype('str').str.zfill(2)
    
    df_rates.loc[:,'value'] = df_bonds['value']**(-1/df_bonds['time2mat'])-1
    df_cash = df_cash.set_index('var', append=True)
    df_rates = df_rates[df_rates['time2mat']>0]
    df_rates = df_rates.set_index('var', append=True)
    df_out = pd.concat([df_cash, df_rates['value']], axis=1)   
    #ESG Format - columns are time to maturity
    df_out.unstack(level='var').dropna(axis=1, how='all').to_clipboard()     
    print("done!")
    
plot_short_rate_dist()
