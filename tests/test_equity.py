#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:44:17 2017

@author: luk-f-a
"""

import unittest
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil import equity_models
from esglil.common import TimeDependentParameter
from esglil.ir_models import HullWhite1fShortRate, HullWhite1fCashAccount, DeterministicBankAccount
from esglil.esg import ESG
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt

#TODO: test delta_t_in and out different than one and different than each other
class gbm_basic_test(unittest.TestCase):
    def setUp(self):
        delta_t = 0.1
        dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])
        S = equity_models.GeometricBrownianMotion(mu=0.02, sigma=0.2, dW=dW)
        self.esg = ESG(dt_sim=delta_t, dW=dW, S=S)

    def test_shape(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        self.assertEqual(type(df_full_run), pd.DataFrame,
                         'incorrect type')
        self.assertEqual(df_full_run.shape, (10*40, 2))
        ax=None
        for sim, s in df_full_run[['S']].groupby(level='sim'):
            if ax is None:
                ax = s.reset_index('sim')['S'].plot()
            else:
                ax = s.reset_index('sim')['S'].plot(ax=ax)
        plt.show()      
        
class gbm_leakage_test(unittest.TestCase):
    def setUp(self):
        delta_t = 0.01
        dW = rng.NormalRng(dims=1, sims=50000, mean=[0], cov=[[delta_t]])
        self.mu = 0.02
        self.sigma = 0.2
        cash = DeterministicBankAccount(r=self.mu)
        S = equity_models.GeometricBrownianMotion(mu=self.mu, sigma=self.sigma, dW=dW)
        self.esg = ESG(dt_sim=delta_t, dW=dW, cash=cash, S=S)

    def test_mean(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        discounted_S = df_full_run['S']/df_full_run['cash']
        mc_mean = discounted_S.groupby(level='time').mean().values
        ref_mean = 100*np.ones((1,40))
        errors = mc_mean/ref_mean-1
        if not np.all(np.less(np.abs(errors), 0.01)):
            print(errors)
        self.assertTrue(np.all(np.less(np.abs(errors), 0.01)))

class gbm_statistical_test_stochastic_r(unittest.TestCase):
    def setUp(self):
        delta_t = 0.01
        rho = 0.2
        corr = [[1, rho], [rho, 1]]
        C = np.diag([delta_t, delta_t])
        cov = C*corr*C.T 
        dW = rng.NormalRng(dims=2, sims=1000, mean=[0, 0], cov=cov)
        a = 0.001
        sigma_r = 0.01
        B = TimeDependentParameter(function=lambda t: 0.01)
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma_r, dW=dW[0])
        C = HullWhite1fCashAccount(r=r)
        S = equity_models.GeometricBrownianMotion(mu=r, sigma=0.2, dW=dW[1])        
        self.esg = ESG(dt_sim=delta_t, dW=dW, r=r, cash=C, S=S)
        
    def test_mean(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        df_full_run['S/cash'] = df_full_run['S']/df_full_run['cash']
        mc_mean = df_full_run['S/cash'].groupby('time').mean().values
        ref_mean = 100*np.ones((1,40))
        errors = mc_mean/ref_mean-1
        if not np.all(np.less(np.abs(errors), 0.01)):
            print(errors)
        self.assertTrue(np.all(np.less(np.abs(errors), 0.01)))

class gbm_test_stochastic_int_rate(unittest.TestCase):
    def setUp(self):
        delta_t = 0.1
        rho = 0.2
        corr = [[1, rho], [rho, 1]]
        C = np.diag([delta_t, delta_t])
        cov = C*corr*C.T 
        dW = rng.NormalRng(dims=2, sims=10, mean=[0, 0], cov=cov)
        a = 0.001
        sigma_r = 0.01
        B = TimeDependentParameter(function=lambda t: 0.01)
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma_r, dW=dW[0])
        S = equity_models.GeometricBrownianMotion(mu=r, sigma=0.2, dW=dW[1])
        self.esg = ESG(dt_sim=delta_t, dW=dW, r=r, S=S)

    def test_shape(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        #print(self.esg.df_value_t())
        self.assertEqual(type(df_full_run), pd.DataFrame,
                         'incorrect type')
        self.assertEqual(df_full_run.shape, (10*40, 4))
        ax=None
        for sim, s in df_full_run[['S']].groupby(level='sim'):
            if ax is None:
                ax = s.reset_index('sim')['S'].plot()
            else:
                ax = s.reset_index('sim')['S'].plot(ax=ax)
        plt.show()
        
if __name__ == '__main__':
    unittest.main()              
            