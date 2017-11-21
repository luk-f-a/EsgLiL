# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:22:52 2017

@author: CHK4436
"""

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil import pipeline
from esglil import ir_models
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import datetime
from scipy.stats import normaltest, shapiro, probplot, kstest
import matplotlib.mlab as mlab

def plot_short_rate_all_sims():
    now_ = datetime.datetime.now()
    r_gen = rng.NormalRng(shape={'svar':1, 'sim':10, 'timestep':5},
                            mean=[0], cov=[[1]])
    HW = ir_models.HullWhite1fModel
    fixed_b = 0.05
    b = lambda x: fixed_b
    r_zero = 0.05
    hw = HW(a=1.01, b=b, sigma=0.02, r_zero=r_zero, 
                  delta_t_in=1, delta_t_out=1)
    
    
    X = r_gen.generate()
    sims = hw.transform(X)
        
    for r in sims.values.squeeze():
        row =  np.insert(r,0,r_zero)
        x = np.insert(sims.coords['time'].values,0,0)
        plt.plot(x, row)
        
def plot_short_rate_pcentiles():
    now_ = datetime.datetime.now()
    sims_nb=10000
    over_samp = 52
    r_gen = rng.NormalRng(shape={'svar':1, 'sim':sims_nb, 'timestep':40*over_samp},
                            mean=[0], cov=[[1]])
    HW = ir_models.HullWhite1fModel
    fixed_b = 0.05
    b = lambda x: fixed_b
    r_zero = 0.05
    hw = HW(a=1.01, b=b, sigma=0.02, r_zero=r_zero, 
                  delta_t_in=1, delta_t_out=10)
    
    
    X = r_gen.generate()
    sims = hw.transform(X)
    print(datetime.datetime.now()-now_)
    x = np.insert(sims.coords['timestep'].values,0,0)
    sims = np.concatenate([np.ones((1,sims_nb,1))*r_zero, sims], axis=2).squeeze()
    pctiles = [np.percentile(sims.squeeze(), p, axis=0) for p in [0,25,75,100]]
    mean = np.median(sims.squeeze(), axis=0)
    plt.plot(x, mean, 'k')
    n = len(pctiles)
    color = plt.cm.Blues(np.linspace(0.1,0.9,n))

    pct_pairs = list(zip(pctiles[:-1], pctiles[1:]))
    for i in range(n//2):
        p1, p2 = pct_pairs[i]
        plt.fill_between(x, p1, p2, facecolor=color[-i], alpha=0.5)
        p1, p2 = pct_pairs[-i-1]
        plt.fill_between(x, p1, p2, facecolor=color[-i], alpha=0.5)
    plt.show()

def plot_short_rate_dist():
    now_ = datetime.datetime.now()
    sims_nb=1_000
    time_sampling_factor = 20
    max_time = 1
    r_gen = rng.NormalRng(shape={'svar':1, 'sim':sims_nb, 'timestep':max_time*time_sampling_factor},
                            mean=[0], cov=[[1/time_sampling_factor]])
    HW = ir_models.HullWhite1fModel
    fixed_b = 0.05
    b = lambda x: fixed_b
    r_zero = 0.05
    sigma = 0.02
    a = 1.01
    hw = HW(a=a, b=b, sigma=sigma, r_zero=r_zero, 
                  delta_t_in=1/time_sampling_factor, delta_t_out=1)
    X = r_gen.generate()
    sims = hw.transform(X)
    print(datetime.datetime.now()-now_)
    test_time = 1
    sims_test = sims.loc[{'timestep':test_time*time_sampling_factor}]
    theoretical_mu = np.exp(-a*test_time)*r_zero+b(test_time)/a*(1-np.exp(-a*test_time))
    theoretical_sigma = np.sqrt(sigma**2/2/a*(1-np.exp(-2*a*test_time)))
    n, bins, patches = plt.hist(sims_test, bins=100,  normed=1)
    y = mlab.normpdf(bins, theoretical_mu, theoretical_sigma)
    plt.plot(bins, y, '--')    
    plt.show()
    print('mean', theoretical_mu, sims_test.values.squeeze().mean())
    print('sigma', theoretical_sigma, sims_test.values.squeeze().std())


    _, p1 = normaltest(sims_test.values.squeeze())
#    z = (sims_test.values.squeeze()-theoretical_mu)/theoretical_sigma
#    _, p3= kstest(z, 'norm',args=(0,1))
    
    print(p1)
    probplot(sims_test.values.squeeze(),
             sparams=(theoretical_mu,theoretical_sigma),fit=False, plot=plt)
    
plot_short_rate_pcentiles()
plot_short_rate_dist()