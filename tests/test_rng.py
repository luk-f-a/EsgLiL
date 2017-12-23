#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:23:19 2017

@author: luk-f-a
"""

import unittest
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil.esg import ESG
import numpy as np
import xarray as xr
import pandas as pd


class Uniform_Rng_test_np(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        delta_t = 0.01
        self.rng = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])

    def test_returns_object(self):
        r_nb = self.rng.generate()
        self.assertEqual(type(r_nb), np.ndarray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (10,),
                         'incorrect shape')


class Wiener_test(unittest.TestCase):
    def setUp(self):
        delta_t = 0.1
        dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])
        W = rng.WienerProcess(dW=dW)
        self.esg = ESG(dt_sim=delta_t, dW=dW, W=W)
            
    def test_returns_object(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        self.assertEqual(type(df_full_run), pd.DataFrame,
                         'incorrect type')
        self.assertEqual(df_full_run.shape, (10, 2*40))
         
class Normal_Rng_test(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.NormalRng(dims=2, sims=50000, mean=[0,0], cov=np.eye(2,2))
        cov = np.array([[1,0.7],[0.7,1]])
        self.corr_norm =  rng.CorrelatedRV(rng=self.rng, input_cov=np.eye(2,2),
                                           target_cov=cov)
        self.esg = ESG(dt_sim=1, ind_dW=self.rng, dW=self.corr_norm)
        
        
    def test_independece(self):
        cf = self.esg.run_multistep_to_pandas(dt_out=1, max_t=5, 
                                  out_vars=['ind_dW'])
        ind_dW_0 = xr.DataArray(cf).unstack('dim_1').sel(model=['ind_dW_0'])
        #print('ind', np.round(np.corrcoef(ind_dW_0.values.squeeze(), rowvar=False),1))        
        self.assertTrue(np.allclose(np.round(np.corrcoef(ind_dW_0.values.squeeze(), rowvar=False),1), np.eye(5,5)))
        
    def test_covariance(self):
        cf = self.esg.run_multistep_to_pandas(dt_out=1, max_t=5, 
                                  out_vars=['dW'])
        dW_0 = xr.DataArray(cf).unstack('dim_1')
        dW_0= dW_0.stack(svar_time=('model', 'time'))
        corr = np.round(np.corrcoef(dW_0.values.squeeze(), rowvar=False),1)
        self.assertTrue(np.allclose(np.diag(corr), np.ones(10)))
        self.assertTrue(np.allclose(np.diag(corr,5), 0.7*np.ones(5)))
        self.assertTrue(np.allclose(np.diag(corr,-5), 0.7*np.ones(5)))
        
class MCMV_Normal_Rng_test(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.MCMVNormalRng(dims=2, sims_outer=50000, sims_inner=100,
                                     mean=[0,0], cov=np.eye(2,2), mcmv_time=1)
        cov = np.array([[1,0.7],[0.7,1]])
        self.corr_norm =  rng.CorrelatedRV(rng=self.rng, input_cov=np.eye(2,2),
                                           target_cov=cov)
        self.esg = ESG(dt_sim=1/2, ind_dW=self.rng, dW=self.corr_norm)
        
    def test_sims(self):
        #_before_mcmv_time
        cf = self.esg.run_multistep_to_pandas(dt_out=1/2, max_t=2, 
                                  out_vars=['ind_dW'])
        shape = cf.xs('ind_dW_0', level='model', axis=1)[0.5].unique().shape
        self.assertEqual(shape[0], self.rng.sims_outer)

        #after_mcmv_time
        shape = cf.xs('ind_dW_0', level='model', axis=1)[1.5].unique().shape
        self.assertEqual(shape[0], self.rng.sims_outer*self.rng.sims_inner)
        
        #test_mean
        mean = cf.xs('ind_dW_0', level='model', axis=1).mean()
        self.assertTrue(np.allclose(mean, 0, atol=0.01))
        
        #np.tile(np.arange(1,6),2).reshape(2,5)

if __name__ == '__main__':
    unittest.main()              
            