#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:49:10 2017

@author: luk-f-a
"""

import unittest
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil.esg import ESG
from esglil.common import TimeDependentParameter
from esglil.ir_models import (HullWhite1fShortRate, HullWhite1fBondPrice, 
                                HullWhite1fCashAccount, hw1f_B_function, 
                                DeterministicBankAccount)
from esglil.equity_models import GeometricBrownianMotion
from esglil import ir_models
import numpy as np
import pandas as pd
from scipy.stats import normaltest, kstest

#TODO: test delta_t_in and out different than one and different than each other

        
class test_model_replacement(unittest.TestCase):
    def setUp(self):
        self.delta_t = 1
        dW = rng.NormalRng(dims=1, sims=50000, mean=[0], cov=[[self.delta_t]])
        a = 0.001
        sigma = 0.01
        B = TimeDependentParameter(function=lambda t: 0.01)
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
        self.esg = ESG(dt_sim=self.delta_t, dW=dW, B=B, r=r)

    def test_before_replacement(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=2, 
                                                       out_vars=['dW'])
        mean = df_full_run.xs('dW', level='model', axis=1).mean()
        self.assertTrue(np.allclose(mean, 0, atol=0.01))
    
    def test_after_replacement(self):
        dW_new = rng.NormalRng(dims=1, sims=50000, mean=[10], cov=[[self.delta_t]])
        self.esg['dW'] = dW_new
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=2, 
                                                       out_vars=['dW'])
        mean = df_full_run.xs('dW', level='model', axis=1).mean()
        self.assertTrue(np.allclose(mean, 10, atol=0.01))
        

class test_MCMV_hw(unittest.TestCase):
    def setUp(self):
        self.delta_t = 1/100
        dW = rng.NormalRng(dims=2, sims=50000, mean=[0,0], 
                           cov=[[self.delta_t,0],[0,self.delta_t]])
        hw_a = 0.01
        hw_sigma = 0.01
        # 
        bond_rate = 0.02
        bond_prices = {i:(1+bond_rate)**(-i) for i in range(1,15)}
        B_fc, f, p = hw1f_B_function(bond_prices, hw_a, hw_sigma, return_p_and_f=True)
        B = TimeDependentParameter(B_fc)
        r = HullWhite1fShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[1])
        P_0 = np.array(list(bond_prices.values())).reshape(-1,1)
        T = np.array(list(bond_prices.keys())).reshape(-1,1)
        P = HullWhite1fBondPrice(a=hw_a, r=r, sigma=hw_sigma, 
                             P_0=p,f=f, T=T)
        P_y10 = HullWhite1fBondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                             P_0=p, f=f, T=10)
        C = HullWhite1fCashAccount(r=r)
        self.esg = ESG(dt_sim=self.delta_t, dW=dW, B=B,r=r, C=C, P=P, P_y10=P_y10)
        
    def test_bonds(self):

        dW_new = rng.MCMVNormalRng(dims=2, sims_outer=5, sims_inner=1000,
                                     mean=[0,0], 
                                     cov=[[self.delta_t,0],[0,self.delta_t]],
                                     mcmv_time=1)
        self.esg['dW'] = dW_new
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=14,
                                                       out_vars=['C', 'P'])
        out = np.tile(np.arange(dW_new.sims_outer), dW_new.sims_inner)
        inn = np.repeat(np.arange(dW_new.sims_inner), dW_new.sims_outer)
        tuples = list(zip(out, inn))
        index = pd.MultiIndex.from_tuples(tuples, names=['outer_sim', 'inner_sim'])
        df_full_run.index = index
        mean_cash = (1/df_full_run.xs('C', level='model', axis=1)).groupby(level='outer_sim').mean()


        c_1 = df_full_run.xs(1, level='time', axis=1)['C'].values
        mean_bond = df_full_run.xs(1, level='time', axis=1).div(c_1, axis='index').groupby(level='outer_sim').mean()
        del mean_bond['C']
        mean_bond.columns.name = 'time'
        mean_bond = mean_bond.rename({'P_'+str(i).zfill(2):i+1 for i in range(15)}, axis=1)
        errors = mean_cash.stack()-mean_bond.stack()
        test = np.allclose(errors, 0, atol=0.01)
        if not test:
             print("Errors in MCMV bond pricing", 
                  errors.abs().groupby(level='time').max())
        self.assertTrue(test)
        resource.setrlimit(rsrc, (8*1024**3, -1))        
        
    def test_1_bond(self):
        """
        Tests 1 bond (10 year one): all terminal values should be one and
        implied rate should be 2%
        """
        dW_new = rng.MCMVNormalRng(dims=2, sims_outer=50, sims_inner=100,
                                     mean=[0,0], 
                                     cov=[[self.delta_t,0],[0,self.delta_t]],
                                     mcmv_time=1)
        self.esg['dW'] = dW_new
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=10,
                                                       out_vars=['C', 'P_y10'])
        out = np.tile(np.arange(dW_new.sims_outer), dW_new.sims_inner)
        inn = np.repeat(np.arange(dW_new.sims_inner), dW_new.sims_outer)
        tuples = list(zip(out, inn))
        index = pd.MultiIndex.from_tuples(tuples, names=['outer_sim', 'inner_sim'])
        df_full_run.index = index
#        mean = [(1/cf.xs('C', level='model', axis=1)[mat]).mean() 
#                for mat in range(1,41)]
        mean_cash = (1/df_full_run.xs('C', level='model', axis=1))[10].groupby(level='outer_sim').mean()
        #print(mean_cash)
        self.assertAlmostEqual(0.02, mean_cash.mean()**(-1/10)-1, places=2)
        terminal_price = df_full_run.xs(10, level='time', axis=1)['P_y10'].groupby(level='outer_sim').mean()
        self.assertTrue(np.allclose(terminal_price, 1, atol=0.0001 ))
        c_1 = df_full_run.xs(1, level='time', axis=1)['C'].values
        mean_bond = df_full_run.xs(1, level='time', axis=1).div(c_1, axis='index').groupby(level='outer_sim').mean()
        self.assertAlmostEqual(0.02, (mean_bond.mean()['P_y10']**(-1/10)-1), places=2)
        
        
        
        
class test_MCMV_gbm(unittest.TestCase):
    def setUp(self):
        self.delta_t = 1/250
        dW = rng.NormalRng(dims=2, sims=10, mean=[0], cov=[[self.delta_t]])
        S = GeometricBrownianMotion(mu=0.02, sigma=0.2, dW=dW[0])
        C = DeterministicBankAccount(r=0.02)
        self.esg = ESG(dt_sim=self.delta_t, dW=dW, S=S, C=C)
        
    def test_index(self):
        dW_new = rng.MCMVNormalRng(dims=2, sims_outer=5, sims_inner=50000,
                                     mean=[0, 0], 
                                     cov=[[self.delta_t,0],[0,self.delta_t]],
                                     mcmv_time=1)
        self.esg['dW'] = dW_new
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=10, 
                                                       out_vars=['C', 'S'])
        out = np.tile(np.arange(dW_new.sims_outer), dW_new.sims_inner)
        inn = np.repeat(np.arange(dW_new.sims_inner), dW_new.sims_outer)
        tuples = list(zip(out, inn))
        index = pd.MultiIndex.from_tuples(tuples, names=['outer_sim', 'inner_sim'])
        df_full_run.index = index

#        mean = [(1/cf.xs('C', level='model', axis=1)[mat]).mean() 
#                for mat in range(1,41)]
        mean_cash = (df_full_run.xs('S', level='model', axis=1)/df_full_run.xs('C', level='model', axis=1)).groupby(level='outer_sim').mean()
#        print(mean_cash)
        c_1 = df_full_run.xs(1, level='time', axis=1)['C'].values
        mean_bond = df_full_run.xs(1, level='time', axis=1).div(c_1, axis='index').groupby(level='outer_sim').mean()
#        print(mean_bond)
        for col in mean_cash.columns.tolist():
            with self.subTest(time=col):
                self.assertTrue(np.allclose(mean_cash[col].values, mean_bond['S'].values, rtol=0.01))
                
if __name__ == '__main__':
    import resource
    rsrc = resource.RLIMIT_AS
    resource.setrlimit(rsrc, (8*1024**3, -1))
    unittest.main()    