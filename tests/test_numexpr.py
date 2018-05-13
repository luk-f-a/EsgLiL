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
from esglil.ir_models.hw1f_euler import ShortRate, CashAccount, BondPrice, B_function
from esglil.ir_models.common import DeterministicBankAccount
from esglil.esg import ESG
import numpy as np
import datetime
import pandas as pd
from matplotlib import pyplot as plt
import numexpr as ne

class gbm_basic_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        delta_t = 1
        np.random.seed(0)
#        print(np.random.normal(0,1,3))
        dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])
        S = equity_models.GeometricBrownianMotion(mu=0.02, sigma=0.2, dW=dW)
        cls.esg = ESG(dt_sim=delta_t, dW=dW, S=S)
       
        cls.S_10 = cls.esg.run_multistep_to_dict(dt_out=1, max_t=10, 
                                              use_numexpr=False)
        cls.S_10 = cls.S_10[10]['S']
        
        np.random.seed(0)
#        print(np.random.normal(0,1,3))
        dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])
        S = equity_models.GeometricBrownianMotion(mu=0.02, sigma=0.2, dW=dW)
        cls.esg = ESG(dt_sim=delta_t, dW=dW, S=S)

        cls.S_10_ne = cls.esg.run_multistep_to_dict(dt_out=1, max_t=10, 
                                                 use_numexpr=True)
        cls.S_10_ne = cls.S_10_ne[10]['S']
        
    def test_ne_np(self):
#        print(self.S_10, self.S_10_ne)
        self.assertTrue(np.array_equal(self.S_10, self.S_10_ne))
       

class numexpr_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ne.set_num_threads(2)
        ### Numpy Run
        np.random.seed(0)        
        delta_t = 0.1
        rho = 0.2
        hw_a = 0.001
        hw_sigma = 0.01
        corr = [[1, rho], [rho, 1]]
        C = np.diag([delta_t, delta_t])
        cov = C*corr*C.T 
        dW = rng.NormalRng(dims=2, sims=5, mean=[0, 0], cov=cov)
        a = 0.001
        sigma_r = 0.01
        B = TimeDependentParameter(function=lambda t: 0.01)
        r = ShortRate(B=B, a=a, sigma=sigma_r, dW=dW[0])
        bond_prices = {t: 1.02**-t for t in range(1,41)}
        T = np.array(list(bond_prices.keys())).reshape(-1,1)
        B_fc, f, p0 = B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
        P = BondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                             P_0=p0, f=f, T=T)
        S = equity_models.GeometricBrownianMotion(mu=r, sigma=0.2, dW=dW[1])
        C = CashAccount(r=r)
        cls.esg = ESG(dt_sim=delta_t, dW=dW, r=r, S=S, C=C, P=P)
        tic = datetime.datetime.now()
        cls.vars_10 = cls.esg.run_multistep_to_dict(dt_out=1, max_t=40, 
                                              use_numexpr=False)
        print('numpy', datetime.datetime.now()-tic)
        cls.vars_10 = cls.vars_10[40]
        
        ### Numexpr Run
        np.random.seed(0)        
        delta_t = 0.1
        rho = 0.2
        corr = [[1, rho], [rho, 1]]
        C = np.diag([delta_t, delta_t])
        cov = C*corr*C.T 
        dW = rng.NormalRng(dims=2, sims=5, mean=[0, 0], cov=cov)
        a = 0.001
        sigma_r = 0.01
        B = TimeDependentParameter(function=lambda t: 0.01)
        r = ShortRate(B=B, a=a, sigma=sigma_r, dW=dW[0])
        bond_prices = {t: 1.02**-t for t in range(1,41)}
        T = np.array(list(bond_prices.keys())).reshape(-1,1)
        B_fc, f, p0 = B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
        P = BondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                             P_0=p0, f=f, T=T)
        S = equity_models.GeometricBrownianMotion(mu=r, sigma=0.2, dW=dW[1])
        C = CashAccount(r=r)
        cls.esg = ESG(dt_sim=delta_t, dW=dW, r=r, S=S, C=C, P=P)
        tic = datetime.datetime.now()
        cls.vars_10_ne = cls.esg.run_multistep_to_dict(dt_out=1, max_t=40, 
                                              use_numexpr=True)
        print('numpy+numexpr', datetime.datetime.now()-tic)
        cls.vars_10_ne = cls.vars_10_ne[40]
        
    def test_compare(self):
        for eq in ['S', 'r', 'C', 'P']: 
            with self.subTest(variable=eq):
                if not np.allclose(self.vars_10[eq], self.vars_10_ne[eq]):
                    print(eq, self.vars_10[eq], self.vars_10_ne[eq])
                    print(self.vars_10[eq]== self.vars_10_ne[eq])
                self.assertTrue(np.allclose(self.vars_10[eq], self.vars_10_ne[eq]))
        
if __name__ == '__main__':
    unittest.main()              
            