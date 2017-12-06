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
from esglil.esg import esg
from esglil.common import TimeDependentParameter
from esglil.ir_models import HullWhite1fShortRate, HullWhite1fBondPrice, HullWhite1fCashAccount
import numpy as np
import pandas as pd
from scipy.stats import normaltest, kstest

#TODO: test delta_t_in and out different than one and different than each other

        
class hw1f_test_short_rate_xr(unittest.TestCase):
    def setUp(self):
        delta_t = 1
        max_t = 5
        dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[1/delta_t]])
    #    a = ConstantParameter(0.2)
    #    sigma = ConstantParameter(0.01)
        a = 0.001
        sigma = 0.01
        B = TimeDependentParameter(function=lambda t: 0.1)
        r = ir_models.HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
        self.esg = ESG(dt_sim=delta_t, dt_out=1, dW=dW, B=B, r=r)

        

        
    def test_shape(self):
        df_full_run = esg.full_run_to_pandas(40)
        self.assertEqual(type(df_full_run), pd.DataFrame,
                         'incorrect type')
        self.assertEqual(df_full_run.shape, (10*5, 3))
        df_full_run[['r']].plot()

class hw1f_leakage_tests(unittest.TestCase):    
    def setUp(self):
        self.time_sampling_factor = k = 252
        self.max_time = T = 40
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':1000, 
                                        'timestep':k},
                                mean=[0], cov=[[1/k]], loop_dim='timestep')
        #self.X = self.rng.generate()
        
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

        B, r_0 =  ir_models.hw1f_B_function(bond_prices, a, sigma)
        HW = ir_models.HullWhite1fModel

        

        self.hw = HW(a=a, B=B, sigma=sigma, r_zero=r_0, 
                      delta_t_in=1/self.time_sampling_factor, delta_t_out=1,
                      bond_prices=bond_prices)
            


                
    def test_bond_prices(self):
        """Test that starting bond prices can be recovered
        """
        import pandas as pd
        sims = []
        for year in range(1, 41):
            X = self.rng.generate()
            sims.append(self.hw.transform(X))
            #print(sims[-1].coords)
        self.sims = xr.concat(sims, dim='timestep')
        self.sims.coords['timestep'] = self.sims.coords['timestep']/self.time_sampling_factor
        df_sims = self.sims.to_dataframe(name='value').reset_index()
        df_bonds = df_sims[df_sims.svar.str.startswith('bond')]
        df_cash = df_sims[df_sims.svar.str.startswith('cash_index')].set_index(['svar', 'sim', 'timestep'])
        mat = df_bonds['svar'].str.split('_').str.get(1).astype('float').astype('int')
        ts = df_bonds['timestep'].astype('int')
        df_bonds.loc[:,'time2mat'] = (mat - ts)
        df_bonds.loc[:,'svar'] = 'rate_'+df_bonds['time2mat'].astype('str').str.zfill(2)
        
        df_bonds.loc[:,'rate'] = df_bonds['value']**(-1/df_bonds['time2mat'])-1
        df_bonds = df_bonds[['svar', 'sim', 'timestep', 'rate']].set_index(['svar', 'sim', 'timestep'])
        df_out = pd.concat([df_cash, df_bonds], axis=1)   
        #ESG Format - columns are time to maturity
        df_out.unstack(['svar']).dropna(axis=1, how='all').to_clipboard()
        #Simulation natural Format - columns are bond maturity
        #self.sims.to_dataframe(name='test').unstack(['svar']).to_clipboard()
        
        
        
#class hw1f_stat_test_short_rate(unittest.TestCase):            
#    def setUp(self):
#        self.time_sampling_factor = k = 10
#        self.max_time = T = 10
#        self.rng = rng.NormalRng(shape={'svar':1, 'sim':100_000, 
#                                        'timestep':T*k},
#                                mean=[0], cov=[[1/k]])
#        self.X = self.rng.generate()
#    def test_fixed_mean_reversion1(self):
#        """Test that starting from a level it stays in that level
#        """
#       
#        HW = ir_models.HullWhite1fModel
#        a = 0.5
#        mean_rev_level = 0.05
#        fixed_b = mean_rev_level*a
#        b = lambda x: fixed_b
#        r_zero = 0.05
#        hw = HW(a=a, b=b, sigma=0.02, r_zero=r_zero, 
#                      delta_t_in=1/self.time_sampling_factor, delta_t_out=1)
#
#        sims = hw.transform(self.X)
#        mean_ = sims.mean(dim='sim').values.squeeze()
#        self.assertTrue(np.allclose(mean_, [mean_rev_level]*10, rtol=0.01))
#
#    def test_fixed_mean_reversion2(self):
#        """Test that starting from a high level it reverts
#        """
#        HW = ir_models.HullWhite1fModel
#        a = 0.5
#        mean_rev_level = 0.05
#        fixed_b = mean_rev_level*a
#        b = lambda x: fixed_b
#        r_zero = 0.30
#        hw = HW(a=a, b=b, sigma=0.02, r_zero=r_zero, 
#                      delta_t_in=1/self.time_sampling_factor, delta_t_out=1)
#
#        sims = hw.transform(self.X)
#        mean_ = sims.mean(dim='sim').values.squeeze()
#        dist = (mean_- np.array([mean_rev_level]*10))
#        diff_dist = np.diff(dist)
#        self.assertTrue(np.all(diff_dist<0))
#        
#    def test_distribution(self):
#        """Test that when starting from a level it stays in that level
#        """
#        k = 5_000
#        T = 1
#        sim_nb = 1000
#        self.rng = rng.NormalRng(shape={'svar':1, 'sim':sim_nb, 
#                                        'timestep':T*k},
#                                mean=[0], cov=[[1/k]])
#        self.X = self.rng.generate()
#        res = True
#        HW = ir_models.HullWhite1fModel
#        a = 0.5
#        mean_rev_level = 0.05
#        fixed_b = mean_rev_level*a
#        b = lambda x: fixed_b
#        r_zero = 0.30
#        sigma = 0.02
#        hw = HW(a=a, b=b, sigma=sigma, r_zero=r_zero, 
#                      delta_t_in=1/k, delta_t_out=1)
#        test_time = 1
#        test_ts = test_time*k
#        sims = hw.transform(self.X)
#        sims_test = sims.loc[{'timestep': test_ts}]
#        theoretical_mu = np.exp(-a*test_time)*r_zero+b(test_time)/a*(1-np.exp(-a*test_time))
#        theoretical_sigma = np.sqrt(sigma**2/2/a*(1-np.exp(-2*a*test_time)))
# 
#        if sim_nb > 1000:
#            _, p1 = normaltest(sims_test.values.squeeze())
#        else:
#            _, p1 = kstest(sims_test.values.squeeze(), 'norm',
#                           args=(theoretical_mu,theoretical_sigma) )
#        if p1 < 0.1:
#            print('Normality Test failed: p-value {}'.format(p1))
#            res = False
#        
#        
#        if round(float(sims_test.mean().squeeze()), 3)!= round(theoretical_mu,3):   
#            a = float(sims_test.mean().squeeze())
#            b = theoretical_mu
#            print('Fail:_Empirical mean {}, closed form mean {}'.format(a, b))            
#            res = False
#        if round(float(sims_test.std().squeeze()), 3)!= round(theoretical_sigma,3):   
#            a = float(sims_test.std().squeeze())
#            b = theoretical_sigma
#            print('Fail: Empirical std dev {}, closed form std dev {}'.format(a, b))            
#            res = False            
#       
#        self.assertTrue(res)
#        
#class hw1f_test_short_rate_and_bonds_xr(unittest.TestCase):
#    def setUp(self):
#        self.rng = rng.NormalRng(shape={'svar':1, 'sim':10, 'timestep':5},
#                                mean=[0], cov=[[1]])
#        HW = ir_models.HullWhite1fModel
#        fixed_b = 0.05
#        b = lambda x: fixed_b
#        self.r_zero = 0.05
#        bond_p = {1: 99, 2:98}
#        self.hw = HW(a=1.01, b=b, sigma=0.02, r_zero=self.r_zero, 
#                      delta_t_in=1, delta_t_out=1,  bond_prices=bond_p)
#        
#    def test_shape(self):
#        X = self.rng.generate()
#        sims = self.hw.transform(X)
#        self.assertEqual(type(sims), xr.DataArray,
#                         'incorrect type')
#        self.assertEqual(sims.shape, (3, 10,5))
#
#
#class pipeline_hw1f_test_xr(unittest.TestCase):
#    """Test pipeline uniform rng with numpy output
#    """
#    def setUp(self):
#        self.rng = rng.NormalRng(shape={'svar':1, 'sim':10, 'timestep':5},
#                                mean=[0], cov=[[1]])
#        HW = ir_models.HullWhite1fModel
#        fixed_b = 0.05
#        b = lambda x: fixed_b
#        self.r_zero = 0.05
#        self.hw = HW(a=1.01, b=b, sigma=0.02, r_zero=self.r_zero, 
#                      delta_t_in=1, delta_t_out=1)
# 
#        self.ppl = pipeline.Pipeline([self.rng, self.hw])
#        
#    def test_returns_object(self):
#        r_nb = self.ppl.generate()
#        self.assertEqual(type(r_nb), xr.DataArray,
#                         'incorrect type')
#        self.assertEqual(r_nb.shape, (1, 10,5))

    
        
if __name__ == '__main__':
    unittest.main()              
            