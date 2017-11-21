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
from esglil import pipeline
from esglil import ir_models
import numpy as np
import xarray as xr
from scipy.stats import normaltest, kstest

#TODO: test delta_t_in and out different than one and different than each other

        
class hw1f_test_short_rate_xr(unittest.TestCase):
    def setUp(self):
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':10, 'timestep':5},
                                mean=[0], cov=[[1]])
        HW = ir_models.HullWhite1fModel
        fixed_b = 0.05
        b = lambda x: fixed_b
        self.r_zero = 0.05
        self.hw = HW(a=1.01, b=b, sigma=0.02, r_zero=self.r_zero, 
                      delta_t_in=1, delta_t_out=1)
        
    def test_shape(self):
        X = self.rng.generate()
        sims = self.hw.transform(X)
        self.assertEqual(type(sims), xr.DataArray,
                         'incorrect type')
        self.assertEqual(sims.shape, (1, 10,5))
#        for r in sims.values.squeeze():
#            row =  np.insert(r,0,self.r_zero)
#            x = np.insert(sims.coords['timestep'].values,0,0)
#            plt.plot(x, row)

class hw1f_stat_test_short_rate(unittest.TestCase):            
    def setUp(self):
        self.time_sampling_factor = k = 10
        self.max_time = T = 10
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':100_000, 
                                        'timestep':T*k},
                                mean=[0], cov=[[1/k]])
        self.X = self.rng.generate()
    def test_fixed_mean_reversion1(self):
        """Test that starting from a level it stays in that level
        """
       
        HW = ir_models.HullWhite1fModel
        a = 0.5
        mean_rev_level = 0.05
        fixed_b = mean_rev_level*a
        b = lambda x: fixed_b
        r_zero = 0.05
        hw = HW(a=a, b=b, sigma=0.02, r_zero=r_zero, 
                      delta_t_in=1/self.time_sampling_factor, delta_t_out=1)

        sims = hw.transform(self.X)
        mean_ = sims.mean(dim='sim').values.squeeze()
        self.assertTrue(np.allclose(mean_, [mean_rev_level]*10, rtol=0.01))

    def test_fixed_mean_reversion2(self):
        """Test that starting from a high level it reverts
        """
        HW = ir_models.HullWhite1fModel
        a = 0.5
        mean_rev_level = 0.05
        fixed_b = mean_rev_level*a
        b = lambda x: fixed_b
        r_zero = 0.30
        hw = HW(a=a, b=b, sigma=0.02, r_zero=r_zero, 
                      delta_t_in=1/self.time_sampling_factor, delta_t_out=1)

        sims = hw.transform(self.X)
        mean_ = sims.mean(dim='sim').values.squeeze()
        dist = (mean_- np.array([mean_rev_level]*10))
        diff_dist = np.diff(dist)
        self.assertTrue(np.all(diff_dist<0))
        
    def test_distribution(self):
        """Test that when starting from a level it stays in that level
        """
        k = 5_000
        T = 1
        sim_nb = 1000
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':sim_nb, 
                                        'timestep':T*k},
                                mean=[0], cov=[[1/k]])
        self.X = self.rng.generate()
        res = True
        HW = ir_models.HullWhite1fModel
        a = 0.5
        mean_rev_level = 0.05
        fixed_b = mean_rev_level*a
        b = lambda x: fixed_b
        r_zero = 0.30
        sigma = 0.02
        hw = HW(a=a, b=b, sigma=sigma, r_zero=r_zero, 
                      delta_t_in=1/k, delta_t_out=1)
        test_time = 1
        test_ts = test_time*k
        sims = hw.transform(self.X)
        sims_test = sims.loc[{'timestep': test_ts}]
        theoretical_mu = np.exp(-a*test_time)*r_zero+b(test_time)/a*(1-np.exp(-a*test_time))
        theoretical_sigma = np.sqrt(sigma**2/2/a*(1-np.exp(-2*a*test_time)))
 
        if sim_nb > 1000:
            _, p1 = normaltest(sims_test.values.squeeze())
        else:
            _, p1 = kstest(sims_test.values.squeeze(), 'norm',
                           args=(theoretical_mu,theoretical_sigma) )
        if p1 < 0.1:
            print('Normality Test failed: p-value {}'.format(p1))
            res = False
        
        
        if round(float(sims_test.mean().squeeze()), 3)!= round(theoretical_mu,3):   
            a = float(sims_test.mean().squeeze())
            b = theoretical_mu
            print('Fail:_Empirical mean {}, closed form mean {}'.format(a, b))            
            res = False
        if round(float(sims_test.std().squeeze()), 3)!= round(theoretical_sigma,3):   
            a = float(sims_test.std().squeeze())
            b = theoretical_sigma
            print('Fail: Empirical std dev {}, closed form std dev {}'.format(a, b))            
            res = False            
       
        self.assertTrue(res)
        
class hw1f_test_short_rate_and_bonds_xr(unittest.TestCase):
    def setUp(self):
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':10, 'timestep':5},
                                mean=[0], cov=[[1]])
        HW = ir_models.HullWhite1fModel
        fixed_b = 0.05
        b = lambda x: fixed_b
        self.r_zero = 0.05
        bond_p = {1: 99, 2:98}
        self.hw = HW(a=1.01, b=b, sigma=0.02, r_zero=self.r_zero, 
                      delta_t_in=1, delta_t_out=1,  bond_prices=bond_p)
        
    def test_shape(self):
        X = self.rng.generate()
        sims = self.hw.transform(X)
        self.assertEqual(type(sims), xr.DataArray,
                         'incorrect type')
        self.assertEqual(sims.shape, (3, 10,5))


class pipeline_hw1f_test_xr(unittest.TestCase):
    """Test pipeline uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':10, 'timestep':5},
                                mean=[0], cov=[[1]])
        HW = ir_models.HullWhite1fModel
        fixed_b = 0.05
        b = lambda x: fixed_b
        self.r_zero = 0.05
        self.hw = HW(a=1.01, b=b, sigma=0.02, r_zero=self.r_zero, 
                      delta_t_in=1, delta_t_out=1)
 
        self.ppl = pipeline.Pipeline([self.rng, self.hw])
        
    def test_returns_object(self):
        r_nb = self.ppl.generate()
        self.assertEqual(type(r_nb), xr.DataArray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (1, 10,5))

    
        
if __name__ == '__main__':
    unittest.main()              
            