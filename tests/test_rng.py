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
import pandas as pd


class Uniform_Rng_test_np(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        delta_t = 0.01
        self.rng = rng.NormalRng(dims=1, sims=10, mean=[0],
                                 cov=np.array([[delta_t]]))

    def test_returns_object(self):
        r_nb = self.rng.generate()

        self.assertEqual(type(r_nb), np.ndarray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (1, 10),
                         'incorrect shape')


class Wiener_test(unittest.TestCase):
    def setUp(self):
        self.delta_t = 0.1
        dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=np.array([[self.delta_t]]))
        W = rng.WienerProcess(dW=dW)
        self.esg = ESG(dt_sim=self.delta_t, dW=dW, W=W)
            
    def test_returns_object(self):
        df_full_run = self.esg.run_multistep_to_pandas(dt_out=1, max_t=40)
        self.assertEqual(type(df_full_run), pd.DataFrame,
                         'incorrect type')
        self.assertEqual(df_full_run.shape, (10, int(2*40)))


class Normal_Rng_test(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.NormalRng(dims=2, sims=50000, mean=[0,0], cov=np.eye(2,2))
        cov = np.array([[1,0.7],[0.7,1]])
        self.corr_norm =  rng.CorrelatedRV(rng=self.rng, input_cov=np.eye(2,2),
                                           target_cov=cov)
        self.esg = ESG(dt_sim=1, ind_dW=self.rng, dW=self.corr_norm)

    def test_independence(self):
        res = self.esg.run_multistep_to_dict(dt_out=1, max_t=5, out_vars=['ind_dW'])
        res_arr = self.esg.dict_res_to_array(res, 'ind_dW', 'var-time')
        corr = np.corrcoef(res_arr[:,:5], rowvar=False)
        self.assertTrue(np.allclose(np.round(corr, 1), np.eye(5, 5)))
        
    def test_covariance(self):
        res = self.esg.run_multistep_to_dict(dt_out=1, max_t=5,
                                  out_vars=['dW'])
        res_arr = self.esg.dict_res_to_array(res, 'dW', 'var-time')
        corr = np.round(np.corrcoef(res_arr, rowvar=False),1)
        self.assertTrue(np.allclose(np.diag(corr), np.ones(10)))
        self.assertTrue(np.allclose(np.diag(corr,5), 0.7*np.ones(5)))
        self.assertTrue(np.allclose(np.diag(corr,-5), 0.7*np.ones(5)))

class Normal_Rng_seed_test(unittest.TestCase):
    def test_seed(self):
        rng1 = rng.NormalRng(dims=2, sims=3, mean=[0,0], cov=np.eye(2,2), seed=0)
        rng2 = rng.NormalRng(dims=2, sims=3, mean=[0, 0], cov=np.eye(2, 2), seed=0)
        rn1 = rng1.generate()
        rn2 = rng2.generate()
        self.assertTrue(np.array_equal(rn1, rn2))


class MCMV_IndWienerIncr_Rng_test(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.MCMVIndWienerIncr(dims=2, sims_outer=50000, sims_inner=100,
                                     mean=0, delta_t=1/2, mcmv_time=1)
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


class IndWienerIncr_Rng_seed_test(unittest.TestCase):
    def test_seed(self):
        rng1 = rng.IndWienerIncr(dims=2, sims=5, mean=0, delta_t=1/2, seed=0)
        rng2 = rng.IndWienerIncr(dims=2, sims=5, mean=0, delta_t=1/2, seed=0)
        esg = ESG(dt_sim=1, rng1=rng1, rng2=rng2)
        res = esg.run_multistep_to_dict(dt_out=1, max_t=5)
        res_arr1 = esg.dict_res_to_array(res, 'rng1', 'var-time')
        res_arr2 = esg.dict_res_to_array(res, 'rng2', 'var-time')
        self.assertTrue(np.array_equal(res_arr1, res_arr2))


class MCMV_IndWienerIncr_Rng_seed_test(unittest.TestCase):
    def test_seed(self):
        rng1 = rng.MCMVIndWienerIncr(dims=2, sims_outer=5, sims_inner=100,
                                     mean=0, delta_t=1/2, mcmv_time=1, seed=0)
        rng2 = rng.MCMVIndWienerIncr(dims=2, sims_outer=5, sims_inner=100,
                                     mean=0, delta_t=1/2, mcmv_time=1, seed=0)
        esg = ESG(dt_sim=1, rng1=rng1, rng2=rng2)
        res = esg.run_multistep_to_dict(dt_out=1, max_t=5)
        res_arr1 = esg.dict_res_to_array(res, 'rng1', 'var-time')
        res_arr2 = esg.dict_res_to_array(res, 'rng2', 'var-time')
        self.assertTrue(np.array_equal(res_arr1, res_arr2))

class MCMV_IndWienerIncr_Rng_dask_test(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.MCMVIndWienerIncr(dims=2, sims_outer=5000, sims_inner=100,
                                     mean=0, delta_t=1/4, mcmv_time=1,
                                     generator='mc-dask', dask_chunks=2)
        cov = np.array([[1,0.7],[0.7,1]])
        self.corr_norm =  rng.CorrelatedRV(rng=self.rng, input_cov=np.eye(2,2),
                                           target_cov=cov)
        self.esg = ESG(dt_sim=1/2, ind_dW=self.rng, dW=self.corr_norm)
        
    def test_sims(self):
        #_before_mcmv_time
        cf = self.esg.run_multistep_to_dict(dt_out=1/4, max_t=0.5, 
                                  out_vars=['ind_dW'])

#        self.assertTrue(cf[0.5]['ind_dW'].chunks==((2,), (250000, 250000)))
        self.assertTrue(cf[0.5]['ind_dW'].shape == (2,5000*100))

        #after_mcmv_time
        cf = self.esg.run_multistep_to_dict(dt_out=1/4, max_t=2, 
                                  out_vars=['ind_dW'])
        self.assertTrue(cf[1.5]['ind_dW'].chunks==((2,), (250000, 250000)))        
        
        #test_mean
        mean = cf[1.5]['ind_dW'].mean().compute()
        self.assertTrue(np.allclose(mean, 0, atol=0.01))


if __name__ == '__main__':
    unittest.main()              
            