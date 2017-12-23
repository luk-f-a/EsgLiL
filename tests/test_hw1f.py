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
from esglil.esg import ESG
from esglil.common import TimeDependentParameter
from esglil.ir_models import HullWhite1fShortRate, HullWhite1fBondPrice, HullWhite1fCashAccount
from esglil import ir_models
import numpy as np
import pandas as pd
from scipy.stats import normaltest, kstest

#TODO: test delta_t_in and out different than one and different than each other

        
class hw1f_test_short_rate(unittest.TestCase):
    def setUp(self):
        delta_t = 1
        dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])
        a = 0.001
        sigma = 0.01
        B = TimeDependentParameter(function=lambda t: 0.01)
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
        self.esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r)

    def test_shape(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        self.assertEqual(type(df_full_run), pd.DataFrame,
                         'incorrect type')
        self.assertEqual(df_full_run.shape, (10*40, 3))
        ax=None
        for sim, r in df_full_run[['r']].groupby(level='sim'):
            if ax is None:
                ax = r.reset_index('sim')['r'].plot()
            else:
                ax = r.reset_index('sim')['r'].plot(ax=ax)

class hw1f_leakage_tests(unittest.TestCase):    
    def setUp(self):

        
        self.bond_prices = {1: 1.0073, 2: 1.0111, 3: 1.0136, 4: 1.0138, 5: 1.0115, 6: 1.0072,
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
        sigma = ir_models.hw1f_sigma_calibration(self.bond_prices, 
                                                    swaption_prices, a)

        B =  ir_models.hw1f_B_function(self.bond_prices, a, sigma)
        delta_t = 1/25
        dW = rng.NormalRng(dims=1, sims=1000, mean=[0], cov=[[delta_t]])
        B = TimeDependentParameter(function=B)
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
        #T = np.array(list(bond_prices.keys()))
        P = {'Bond_{}'.format(i):HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW, 
                                 P_0=self.bond_prices[i], T=i) for i in range(1,41)}
        C = HullWhite1fCashAccount(r=r)
        self.esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r, cash=C, **P)
            
    def test_cash_to_initial_bond_prices(self):
        """Test that starting bond prices can be recovered
        """
        
        df_sims = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        df_sims = df_sims.swaplevel(0,1)
        df_sims = df_sims.stack()
        df_sims.name = 'value'
        df_sims.index.names = ['sim','time', 'var']
        df_sims = df_sims.reset_index('var')
        df_bonds = df_sims[df_sims['var'].str.startswith('Bond_')]
        df_sims = df_sims.set_index('var', append=True)
        df_cash = df_sims.xs('cash', level='var')
        df_cash = 1/df_cash
        mc_bond_prices = df_cash.groupby('time').mean()
        errors = mc_bond_prices.values / pd.DataFrame.from_dict(self.bond_prices, orient='index').values -1 
        self.assertTrue(np.all(np.less(np.abs(errors), 0.01)))

        
class hw1f_stat_test_short_rate(unittest.TestCase):            
    def setUp(self):
       
        self.delta_t = 0.1
        self.dW = rng.NormalRng(dims=1, sims=100_000, mean=[0], cov=[[self.delta_t]])
        
    def test_fixed_mean_reversion1(self):
        """Test that starting from a level it stays in that level
        """
        a = 0.5
        mean_rev_level = 0.05
        B = TimeDependentParameter(function=lambda t: mean_rev_level)
        sigma = 0.01
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=self.dW)
        self.esg = ESG(dt_sim=self.delta_t, dW=self.dW, B=B, r=r)
        self.df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=10)
       

        mean_ = self.df_full_run[['r']].groupby('time').mean().values
        self.assertTrue(np.allclose(mean_, [mean_rev_level]*10, rtol=0.01))

    def test_fixed_mean_reversion2(self):
        """Test that starting from a high level it reverts
        """
        a = 0.5
        mean_rev_level = 0.05
        B_fc = lambda t: mean_rev_level/(t+0.1)
        B = TimeDependentParameter(function= B_fc)
        sigma = 0.01
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=self.dW)
        self.esg = ESG(dt_sim=self.delta_t, dW=self.dW, B=B, r=r)
        self.df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=10)
       

        mean_ = self.df_full_run[['r']].groupby('sim').mean().values
        ref_mean = np.array([B_fc(t) for t in range(1, 11)])
        self.assertTrue(np.allclose(mean_.T, ref_mean, rtol=0.1))
        
    def test_distribution(self):
        """Test r follows the distribution implied by the solution of its SDE
        """
        res = True
        self.delta_t = 1/5000
        sim_nb = 1000
        self.dW = rng.NormalRng(dims=1, sims=sim_nb, mean=[0], cov=[[self.delta_t]])
        T=1
        a = 0.5
        mean_rev_level = 0.05
        sigma = 0.01

        B_fc = lambda t: mean_rev_level + 0.1*t
        B = TimeDependentParameter(function= B_fc)
        b = lambda t: 0.1 #derivative of B_fc
        r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=self.dW)
        self.esg = ESG(dt_sim=self.delta_t, dW=self.dW, B=B, r=r)
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=T)['r']
        
        #theoretical_mu = np.exp(-a*T)*r_zero+b(T)/a*(1-np.exp(-a*T))
        theoretical_mu = B_fc(T)
        theoretical_sigma = np.sqrt(sigma**2/2/a*(1-np.exp(-2*a*T)))
 
        if sim_nb > 1000:
            _, p1 = normaltest(df_full_run.values.squeeze())
        else:
            _, p1 = kstest(df_full_run.values.squeeze(), 'norm',
                           args=(theoretical_mu,theoretical_sigma) )
        if p1 < 0.1:
            print('Normality Test failed: p-value {}'.format(p1))
            res = False
        

        a = float(df_full_run.groupby('time').mean().squeeze())
        b = theoretical_mu        
        if round(a, 3)!= round(b,3):   
            print('Fail:_Empirical mean {}, closed form mean {}'.format(a, b))            
            res = False
        a = df_full_run.std()
        b = theoretical_sigma
        if round(a, 3)!= round(b, 3):   
            a = float(df_full_run.std().squeeze())
            b = theoretical_sigma
            print('Fail: Empirical std dev {}, closed form std dev {}'.format(a, b))            
            res = False            
       
        self.assertTrue(res)
   

    
        
if __name__ == '__main__':
    unittest.main()              
            