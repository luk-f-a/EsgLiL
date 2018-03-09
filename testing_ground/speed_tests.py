#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:23:35 2017

@author: luk-f-a
"""

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil.esg import ESG
from esglil.common import TimeDependentParameter
from esglil import equity_models
from esglil.ir_models import (HullWhite1fShortRate, HullWhite1fBondPrice, 
                              HullWhite1fCashAccount, get_HWyearly_g_fc,
                              HWyearlyShortRate)

import numpy as np
import pandas as pd
import datetime
from itertools import product, chain


def hw1f_test_short_rate(dt=1, sims=1000):
    delta_t = dt
    dW = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[delta_t]])
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.01)
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r)
    df_full_run = esg.full_run(dt_out=1, max_t=40)

def hw1f_test_short_rate_bonds_cash(dt=1, sims=1000):
    delta_t = dt
    dW = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[delta_t]])
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.01)
    bond_prices = {i:1 for i in range(1,41)}
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    P = {'Bond_{}'.format(i):HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
                             P_0=bond_prices[i], T=i) for i in range(1,41)}
   
    C = HullWhite1fCashAccount(r=r)
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r, cash=C, **P)
    esg.run_step()
    full_run = esg.full_run(dt_out=1, max_t=40)


def hw1f_test_short_rate_vector_bonds_cash(dt=1, sims=1000):
    delta_t = dt
    dW = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[delta_t]])
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.01)
#    bond_prices = {i:1 for i in range(1,41)}
#    P = {'Bond_{}'.format(i):HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
#                             P_0=bond_prices[i], T=i) for i in range(1,41)}
    bond_prices = np.ones(shape=(40,1))
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    T = np.arange(1,41).reshape(-1,1)
    P = HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
                             P_0=bond_prices, T=T)
    
    C = HullWhite1fCashAccount(r=r)
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r, cash=C, P=P)
    full_run = esg.full_run(dt_out=1, max_t=40)

def hw1f_test_nested_esg_short_rate_vector_bonds_cash(dt=1, sims=1000):
    delta_t = dt
    dW = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[delta_t]])
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.01)
#    bond_prices = {i:1 for i in range(1,41)}
#    P = {'Bond_{}'.format(i):HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
#                             P_0=bond_prices[i], T=i) for i in range(1,41)}
    bond_prices = np.ones(shape=(40,1))
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    T = np.arange(1,41).reshape(-1,1)
    P = HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
                             P_0=bond_prices, T=T)
    
    C = HullWhite1fCashAccount(r=r)
    C2 = HullWhite1fCashAccount(r=r)
    esg1 = ESG(dt_sim=delta_t, dW=dW, B=B, r=r, cash=C, P=P)
    esg2 = ESG(dt_sim=1, esg1=esg1, cash2=C2)
    full_run = esg2.full_run(dt_out=1, max_t=40)

def equity_stochastic_interest_rate(dt=1, sims=1000):
    delta_t = dt
    rho = 0.2
    corr = [[1, rho], [rho, 1]]
    C = np.diag([delta_t, delta_t])
    cov = C*corr*C.T 
    dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=cov)
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.01)
#    bond_prices = {i:1 for i in range(1,41)}
#    P = {'Bond_{}'.format(i):HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
#                             P_0=bond_prices[i], T=i) for i in range(1,41)}
    bond_prices = np.ones(shape=(40,1))#*np.ones(shape=(1,sims))
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW[0])
    T = np.arange(1,41).reshape(-1,1)
    P = HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW[0], 
                             P_0=bond_prices, T=T)
    
    C = HullWhite1fCashAccount(r=r)
    S = equity_models.GeometricBrownianMotion(mu=r, sigma=0.2, dW=dW[1])
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r, cash=C, P=P, S=S)
    
    full_run = esg.full_run(dt_out=1, max_t=40)    

def hw_yearly_short_rate(sims=1000):

    Z = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[1]])

    alpha = 0.1
    sigma = 0.01
    b = 0.01
    now_ = datetime.datetime.now()
    g = get_HWyearly_g_fc(b_s=lambda t:b, t_points=range(0,10), 
                          T_points=range(1,11), alpha=alpha)
#    print(datetime.datetime.now()-now_)
    mu = lambda t: alpha*g[(t, t+1)]
    r = HWyearlyShortRate(mu_r=mu, sigma_hw=sigma, 
                          alpha_hw=alpha, r_zero=b, Z=Z)
    esg = ESG(dt_sim=1, Z=Z, r=r)
    df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=10)
   

    mean_ = df_full_run[['r']].mean(axis=0).values
   
        
        
        
def time_fc(fc, params):
    took = datetime.datetime.now()-datetime.datetime.now()
    for _ in range(3):
        now_ = datetime.datetime.now()
        fc(*params)
        took += datetime.datetime.now()-now_
    return took/3

def time_surface_fc(fc, params):
    results = {}
    for p in params:
        results[tuple(p)]=time_fc(fc, p)
    print(str(datetime.datetime.now()), results)
    

#print(time_fc(hw1f_test_short_rate_bonds_cash, (1/250,1000)))
#print(time_fc(hw1f_test_short_rate_vector_bonds_cash, (1/250,1000)))
#print(time_fc(hw1f_test_nested_esg_short_rate_vector_bonds_cash, (1/250,1000)))
#print(time_fc(equity_stochastic_interest_rate, (1/250,1000)))
#time_surface_fc(hw1f_test_short_rate_vector_bonds_cash, 
#                params= chain( product([1,0.1], [1000,10000]),
#                              [(1/250, 50_000)]))
print(time_fc(hw_yearly_short_rate, (100000,)))
"""
History

    '2017-12-08 15:07:15.871460'
        time_surface_fc(hw1f_test_short_rate, 
                        params= chain( product([1,0.1], [1000,10000]),
                                      [(1/250, 50_000)]))
        {(1, 1000)    : datetime.timedelta(0, 0, 26322), 
        (1, 10000)    : datetime.timedelta(0, 0, 44904), 
        (0.1, 1000)   : datetime.timedelta(0, 0, 105836), 
        (0.1, 10000)  : datetime.timedelta(0, 0, 246273), 
        (0.004, 50000): datetime.timedelta(0, 25, 664423)}

     
        time_surface_fc(hw1f_test_short_rate_bonds_cash, 
                params= chain( product([1,0.1], [1000,10000]),
                              [(1/250, 50_000)]))
        {(1, 1000)    : datetime.timedelta(0, 0, 86121), 
        (1, 10000)    : datetime.timedelta(0, 0, 233020), 
        (0.1, 1000)   : datetime.timedelta(0, 0, 491065), 
        (0.1, 10000)  : datetime.timedelta(0, 1, 197795), 
        (0.004, 50000): datetime.timedelta(0, 104, 939920)}
        
        print(time_fc(hw1f_test_short_rate_bonds_cash, (1/250,1000)))
        0:00:08.360832
        print(time_fc(hw1f_test_short_rate_vector_bonds_cash, (1/250,1000)))
        0:00:03.862875
        print(time_fc(hw1f_test_nested_esg_short_rate_vector_bonds_cash, (1/250,1000)))
        0:00:03.793345
        
    '2017-12-08 18:22'
        After changing rng to keep the same matrix in place for value_t and copying into it every time
        print(time_fc(hw1f_test_short_rate_bonds_cash, (1/250,1000)))
        print(time_fc(hw1f_test_short_rate_vector_bonds_cash, (1/250,1000)))
        print(time_fc(hw1f_test_nested_esg_short_rate_vector_bonds_cash, (1/250,1000)))

        0:00:08.979423
        0:00:04.106385
        0:00:03.924141

    '2017-12-08 18:35'
        After using Variable Slice to deal with partial use
        print(time_fc(hw1f_test_short_rate_bonds_cash, (1/250,1000)))
        print(time_fc(hw1f_test_short_rate_vector_bonds_cash, (1/250,1000)))
        print(time_fc(hw1f_test_nested_esg_short_rate_vector_bonds_cash, (1/250,1000)))
        
        0:00:08.481017
        0:00:03.904816
        0:00:03.723691
        
        Conclusion: indirection of Variable Slice hurts performance a bit
        but less than having to create copies at every step
        
    2017-12-08 18:57:00.990417 
        time_surface_fc(hw1f_test_short_rate_bonds_cash, 
                params= chain( product([1,0.1], [1000,10000]),
                              [(1/250, 50_000)]))
        {(1, 1000)   : datetime.timedelta(0, 0, 37164), 
        (1, 10000)   : datetime.timedelta(0, 0, 142309), 
        (0.1, 1000)  : datetime.timedelta(0, 0, 352957), 
        (0.1, 10000) : datetime.timedelta(0, 1, 90351), 
        (0.004, 50000): datetime.timedelta(0, 118, 629517)}     
        
        time_surface_fc(hw1f_test_short_rate_vector_bonds_cash, 
                params= chain( product([1,0.1], [1000,10000]),
                              [(1/250, 50_000)]))
        {(1, 1000)   : datetime.timedelta(0, 0, 17183), 
        (1, 10000)   : datetime.timedelta(0, 0, 93269), 
        (0.1, 1000)  : datetime.timedelta(0, 0, 162436), 
        (0.1, 10000) : datetime.timedelta(0, 0, 764385), 
        (0.004, 50000): datetime.timedelta(0, 94, 496059)}
        

"""