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
from esglil.model_zoo import esg_e_sr_bonds_cash, get_gbm_hw_2levels, esg_equity
from esglil.ir_models import DeterministicBankAccount as Cash
from esglil.esg import Model

parent = os.path.dirname
sys.path.append(os.path.join(parent(parent(os.getcwd())), 'DrpLiL'))
from drplil.pricing_formulas import BlackScholesEuropeanPrice as BSprice
import numpy as np
import dask as da

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

""" Set 1 """
set_1_summary = "Compare impact of amount of chunks without using multistep"

        
def benchmark_1():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = esg_e_sr_bonds_cash(delta_t=delta_t, sims=5_00_000, 
                                rho=0.2, bond_prices=bp)
    #df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=40)
    for _ in range(50):
        esg.run_step()
        
    out = esg.value_t['cash']
    return (out.mean())
    
def test_array_1a():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = esg_e_sr_bonds_cash(delta_t=delta_t, sims=None, 
                                rho=0.2, bond_prices=bp)
    new_dW = rng.IndWienerIncr(dims=2, sims=5_00_000, delta_t=delta_t, 
                                         use_dask=True, use_cores=2)
    esg['ind_dW'] = new_dW
    for _ in range(50):
        esg.run_step()
        
    out = esg.value_t['cash']
    return (out.mean().compute())    #.mean().compute()
    

def test_array_1b():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = esg_e_sr_bonds_cash(delta_t=delta_t, sims=None, 
                                rho=0.2, bond_prices=bp)
    new_dW = rng.IndWienerIncr(dims=2, sims=5_00_000, delta_t=delta_t, 
                                         use_dask=True, use_cores=1)
    esg['ind_dW'] = new_dW
    for _ in range(50):
        esg.run_step()
        
    out = esg.value_t['cash']
    return (out.mean().compute())    #.mean().compute()

set_1 = [(benchmark_1 , 'np'), 
          (test_array_1a, 'Testing with two chunks'), 
           (test_array_1b, 'Testing with one chunk' )]
"""
On Windows:
Compare impact of amount of chunks without using multistep
np 0:00:53.648000
1.10862349056
Testing with two chunks 0:00:06.502000
1.10863769321
Testing with one chunk 0:00:05.484000
1.10862266693

Why is dask so much faster and why is two chunks slower? Both dask
runs used more than one core.

"""

""" Set 2 """
set_2_summary = "Compare impact of using multistep to pandas and to dict"

def benchmark_2a():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t,
                             sims=1_000_000, 
                                rho=0.2, bond_prices=bp)
    for _ in range(50):
        esg.run_step()
           
    out = esg.value_t['S']
    return (out.mean())
    
def benchmark_2b():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t,
                             sims=1_000_000, 
                                rho=0.2, bond_prices=bp)
    df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=5, out_vars=['S'])
        
    out = df_full_run['S'][5]
    return (out.mean())

def benchmark_2c():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t,
                             sims=1_000_000, 
                                rho=0.2, bond_prices=bp)
    dict_full_run = esg.run_multistep_to_dict(dt_out=1, max_t=5, out_vars=['S'])
    out = dict_full_run[5]['S']
    return (out.mean())

def test_array_2a():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp, use_dask=True, 
                                use_cores=1)
    for _ in range(50):
        esg.run_step()
           
    out = esg.value_t['S']
    return (out.mean().compute())    #.mean().compute()
    
def test_array_2b():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp, use_dask=True, 
                                use_cores=1)
    df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=5, out_vars=['S'])
        
    out = df_full_run['S'][5]
    return (out.mean())    #.mean().compute()

def test_array_2c():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t, delta_t_l2=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp, use_dask=True, 
                                use_cores=1)
    dict_full_run = esg.run_multistep_to_dict(dt_out=1, max_t=5, out_vars=['S'])
    out = dict_full_run[5]['S']
    return (out.mean().compute())    #.mean().compute()

set_2 = [(benchmark_2a , 'np with run_step'), 
         (benchmark_2b , 'np with multistep pandas'), 
         (benchmark_2c , 'np with multistep dict'), 
         (test_array_2a, 'Dask with run step'), 
         (test_array_2b, 'Dask with multistep pandas'), 
         (test_array_2c, 'Dask with multistep dict')]

"""
Results on Windows
Compare impact of using multistep to pandas and to dict
np with run_step 0:00:09.593959
111.389308186
np with multistep pandas 0:00:11.440144
111.38930818588706
np with multistep dict 0:00:10.603061
111.389308186
Dask with run step 0:00:05.900590
111.322432108
Dask with multistep pandas 0:00:17.141714
111.38217598873078
Dask with multistep dict 0:00:05.867586
111.420973418

With Dask, the pandas approach is really impacting performance. This is caused
by not setting the time filter in the out_var inputs. In the next set, I
use a filter and the performance is comparable.

"""

""" Set 3 """
set_3_summary = ("Compare impact of using multistep to pandas and to dict"
                "on more complex output- S plus dW")

    
def benchmark_3b():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t,
                             sims=1_000_000, 
                                rho=0.2, bond_prices=bp)
    df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=5, 
                                              out_vars={'ind_dW': slice(1),
                                                        'S':slice(5,5)})
    df_full_run['ind_dW_0'].sum().sum()
    out = df_full_run['S'][5]
    return (out.mean())

def benchmark_3c():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t,
                             sims=1_000_000, 
                                rho=0.2, bond_prices=bp)
    dict_full_run = esg.run_multistep_to_dict(dt_out=1, max_t=5, 
                                              out_vars={'ind_dW': slice(1),
                                                        'S':slice(5,5)})
    
    dict_full_run[1]['ind_dW'].sum()
        
    out = dict_full_run[5]['S']
    return (out.mean())

    
def test_array_3b():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp, use_dask=True, 
                                use_cores=1)
    df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=5, 
                                               out_vars={'ind_dW': slice(1),
                                                        'S':slice(5,5)})
    df_full_run['ind_dW_0'].sum().sum()
    out = df_full_run['S'][5]
    return (out.mean())    #.mean().compute()

def test_array_3c():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t, delta_t_l2=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp, use_dask=True, 
                                use_cores=1)
    dict_full_run = esg.run_multistep_to_dict(dt_out=1, max_t=5, 
                                               out_vars={'ind_dW': slice(1),
                                                        'S':slice(5,5)})

    
    dict_full_run[1]['ind_dW'].sum().compute()
    out = dict_full_run[5]['S']
    return (out.mean().compute())    #.mean().compute()

set_3 = [
         (benchmark_3b , 'np with multistep pandas'), 
         (benchmark_3c , 'np with multistep dict'), 
         
         (test_array_3b, 'Dask with multistep pandas'), 
         (test_array_3c, 'Dask with multistep dict')]

"""
Compare impact of using multistep to pandas and to dicton more complex output- S plus dW
np with multistep pandas 0:00:09.875987
111.38930818588706
np with multistep dict 0:00:09.827983
111.389308186
Dask with multistep pandas 0:00:06.551655
111.35166018481087
Dask with multistep dict 0:00:06.339634
111.309225143

With timefilter the pandas approach is not so bad with dask.
"""

""" Set 4 """
set_4_summary = ("Compare impact of using multistep to pandas and to dict"
                "on more complex output- Bond prices")

    
def benchmark_4b():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t,
                             sims=1_000_000, rho=0.2, bond_prices=bp,
                             const_tau=np.array([10, 20]))
    df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=5, 
                                              out_vars={'ind_dW': slice(1),
                                                        'constP':slice(None)})
    df_full_run['ind_dW_0'].sum().sum()
    out = df_full_run['constP_10.0'].sum()
    return (out.mean())

def benchmark_4c():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t,
                             sims=1_000_000, rho=0.2, bond_prices=bp,
                             const_tau=np.array([10, 20]))
    dict_full_run = esg.run_multistep_to_dict(dt_out=1, max_t=5, 
                                              out_vars={'ind_dW': slice(1),
                                                        'constP':slice(None)})
    
    dict_full_run[1]['ind_dW'].sum()
    out=0
    for t in dict_full_run:
        out += dict_full_run[t]['constP_10.0'].sum()
    return (out/5)

    
def test_array_4b():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t,delta_t_l2=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp,
                             const_tau=np.array([10, 20]), use_dask=True, 
                                use_cores=1)
    df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=5, 
                                              out_vars={'ind_dW': slice(1),
                                                        'constP':slice(None)})
    df_full_run['ind_dW_0'].sum().sum()
    out = df_full_run['constP_10.0'].sum()
    return (out.mean())    #.mean().compute()

def test_array_4c():
    np.random.seed(0)
    delta_t = 0.1
    bp = {i:1.02**(-i) for i in range(1,41)}
    esg = get_gbm_hw_2levels(delta_t_l1=delta_t, delta_t_l2=delta_t, sims=1_000_000, 
                                rho=0.2, bond_prices=bp,
                             const_tau=np.array([10, 20]), use_dask=True, 
                                use_cores=1)
    dict_full_run = esg.run_multistep_to_dict(dt_out=1, max_t=5, 
                                              out_vars={'ind_dW': slice(1),
                                                        'constP':slice(None)})
    dict_full_run[1]['ind_dW'].sum()
    out=0
    for t in dict_full_run:
        out += dict_full_run[t]['constP_10.0'].sum()
    return (out/5).compute()

set_4 = [
         (benchmark_4b , 'np with multistep pandas'), 
         (benchmark_4c , 'np with multistep dict'), 
         
         (test_array_4b, 'Dask with multistep pandas'), 
         (test_array_4c, 'Dask with multistep dict')]

"""
Windows:
    Compare impact of using multistep to pandas and to dicton more complex output- Bond prices
    np with multistep pandas 0:00:13.880388
    815817.4249968475
    np with multistep dict 0:00:14.399440
    815817.424997
    Dask with multistep pandas 0:00:33.620362
    815954.2565600723
    Dask with multistep dict 0:00:06.139614
    815863.332089

"""

""" Set 5 """
set_5_summary = ("Compare impact of using multistep to pandas and to dict"
                "on more complex output- equity call")

    
def benchmark_5():
    np.random.seed(0)
    dt_out = 1
    r = 0.03
    dt_sim = 0.1
    val_calc_time = 1
    max_t = 5
    sims = 1_000_000
    esg = esg_equity(delta_t=dt_sim, sims=sims, r=r, gbm_sigma=0.2, 
                     use_dask=False)
    
    eur_call = BSprice(S=esg['S'], K=100, T=max_t, r=r, sigma=0.2)
    cash = Cash(r=r)
    model = Model(dt_sim=dt_out, esg=esg, eur_call=eur_call, cash=cash)

    now_ = datetime.datetime.now()
    cf = model.run_multistep_to_pandas(dt_out=dt_out, max_t=max_t, 
                                  out_vars=['W', 'S', 'eur_call', 'cash'])

    
    out = cf['S'].sum()
    return (out.mean())

def test_array_5a():
    np.random.seed(0)
    dt_out = 1
    r = 0.03
    dt_sim = 0.1
    val_calc_time = 1
    max_t = 5
    sims = 1_000_000
    esg = esg_equity(delta_t=dt_sim, sims=sims, r=r, gbm_sigma=0.2, 
                     use_dask=True)
    
#    eur_call = BSprice(S=esg['S'], K=100, T=max_t, r=r, sigma=0.2)
    cash = Cash(r=r)
#    model = Model(dt_sim=dt_out, esg=esg, eur_call=eur_call, cash=cash)
    model = Model(dt_sim=dt_out, esg=esg, cash=cash)
    now_ = datetime.datetime.now()
#    cf = model.run_multistep_to_pandas(dt_out=dt_out, max_t=max_t, 
#                                  out_vars=['W', 'S', 'eur_call', 'cash'])
    cf = model.run_multistep_to_dict(dt_out=dt_out, max_t=max_t, 
                                  out_vars=['W', 'S'])
    
    out = cf[5]['S'].sum()
    return (out.mean().compute())

def test_array_5b():
    np.random.seed(0)
    dt_out = 1
    r = 0.03
    dt_sim = 0.1
    val_calc_time = 1
    max_t = 5
    sims = 1_000_000
    esg = esg_equity(delta_t=dt_sim, sims=sims, r=r, gbm_sigma=0.2, 
                     use_dask=True)
    
    eur_call = BSprice(S=esg['S'], K=100, T=max_t, r=r, sigma=0.2)
    cash = Cash(r=r)
    model = Model(dt_sim=dt_out, esg=esg, eur_call=eur_call, cash=cash)
    now_ = datetime.datetime.now()
    cf = model.run_multistep_to_dict(dt_out=dt_out, max_t=max_t, 
                                  out_vars=['W', 'S', 'eur_call', 'cash'])
    
    print(type(cf[5]['eur_call']))
    out = cf[5]['S'].sum()
    return (out.mean().compute())

def test_array_5c():
    np.random.seed(0)
    dt_out = 1
    r = 0.03
    dt_sim = 0.1
    val_calc_time = 1
    max_t = 5
    sims = 1_000_000
    esg = esg_equity(delta_t=dt_sim, sims=sims, r=r, gbm_sigma=0.2, 
                     use_dask=True)
    
    eur_call = BSprice(S=esg['S'], K=100, T=max_t, r=r, sigma=0.2)
    cash = Cash(r=r)
    model = Model(dt_sim=dt_out, esg=esg, eur_call=eur_call, cash=cash)
    now_ = datetime.datetime.now()
    cf = model.run_multistep_to_dict(dt_out=dt_out, max_t=max_t, 
                                  out_vars=['W', 'S', 'eur_call', 'cash'])

    out = cf[5]['eur_call'].sum()
    return (out.mean().compute())

set_5 = [
         (benchmark_5 , 'np with multistep pandas'),
         (test_array_5a, 'Dask (dict) without call in model'), 
         (test_array_5b, 'Dask (dict) with call in model not using it'),
         (test_array_5c, 'Dask (dict) with call in model and using it')]
    
    

run_set_number = 5
print(eval('set_{}_summary'.format(run_set_number)))
for fc, summary in eval('set_'+str(run_set_number)):
    now_ = datetime.datetime.now()
    res = fc()
    print(summary, datetime.datetime.now()-now_)
    print(res)
    
    
#test_array_2a()
#print('dask a', datetime.datetime.now()-now_)
#
#now_ = datetime.datetime.now()
#test_array_2b()
#print('dask b', datetime.datetime.now()-now_)
##
##
#now_ = datetime.datetime.now()
#benchmark_2()
#print('np', datetime.datetime.now()-now_)