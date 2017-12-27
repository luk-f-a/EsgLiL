#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:46:07 2017

@author: small_screen
"""

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil import equity_models
from esglil.esg import ESG
import datetime
from esglil.model_zoo import esg_e_sr_bonds_cash

def benchmark():
    delta_t = 0.1
    dW = rng.IndependentWienerIncrements(dims=1, sims=5_000_000, delta_t=delta_t, 
                                         distributed=False)
    S = equity_models.GeometricBrownianMotion(mu=0.02, sigma=0.2, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, S=S)
    #df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=40)
    for _ in range(50):
        esg.run_step()
        
    out = esg.value_t['S']
    print(out.mean())
    
def test_array():
    delta_t = 0.1
    dW = rng.IndependentWienerIncrements(dims=1, sims=5_000_000, delta_t=delta_t, 
                                         distributed=True)
    S = equity_models.GeometricBrownianMotion(mu=0.02, sigma=0.2, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, S=S)
    #df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=40)
    for _ in range(50):
        esg.run_step()
        
    out = esg.value_t['S']
    print(out.mean().compute())

#now_ = datetime.datetime.now()
#test_array()
#print('dask', datetime.datetime.now()-now_)
#
#now_ = datetime.datetime.now()
#benchmark()
#print('np', datetime.datetime.now()-now_)

def benchmark_1():
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = esg_e_sr_bonds_cash(delta_t=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp)
    #df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=40)
    for _ in range(50):
        esg.run_step()
        
    out = esg.value_t['S']
    print(out.mean())
    
def test_array_1():
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = esg_e_sr_bonds_cash(delta_t=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp)
    new_dW = rng.IndependentWienerIncrements(dims=2, sims=1_000_000, delta_t=delta_t, 
                                         distributed=True)
    esg['ind_dW'] = new_dW
    for _ in range(50):
        esg.run_step()
        
    out = esg.value_t['P']
    print(out.mean().compute())    #.mean().compute()
    
    
now_ = datetime.datetime.now()
test_array_1()
print('dask', datetime.datetime.now()-now_)

now_ = datetime.datetime.now()
benchmark_1()
print('np', datetime.datetime.now()-now_)