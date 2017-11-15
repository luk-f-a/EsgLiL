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
from esglil import equity_models
import numpy as np
import xarray as xr

#TODO: test delta_t_in and out different than one and different than each other
class equity_test_np(unittest.TestCase):
    def setUp(self):
        self.rng = rng.NormalRng(shape=(10,5),mean=[0], cov=[[1]])
        GBM = equity_models.GeometricBrownianMotion
        self.gbm = GBM(mu=0, sigma=0.2, s_zero=100, delta_t_in=1, delta_t_out=1)
        
    def test_shape(self):
        X = self.rng.generate()
        sims = self.gbm.transform(X)
        self.assertEqual(type(sims), np.ndarray,
                         'incorrect type')
        self.assertEqual(sims.shape, (10,5))
        
class equity_test_xr(unittest.TestCase):
    def setUp(self):
        self.rng = rng.NormalRng(shape={'svar':2, 'sim':10, 'time':5},
                                mean=[0,0], cov=[[1,0],[0,1]])
        GBM = equity_models.GeometricBrownianMotion
        self.gbm = GBM(mu=0, sigma=0.2, s_zero=100, delta_t_in=1, delta_t_out=1)
        
    def test_shape(self):
        X = self.rng.generate()
        sims = self.gbm.transform(X)
        self.assertEqual(type(sims), xr.DataArray,
                         'incorrect type')
        self.assertEqual(sims.shape, (2, 10,5))
        

class pipeline_gbm_test_np(unittest.TestCase):
    """Test pipeline uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.NormalRng(shape=(10,5),mean=[0], cov=[[1]])
        GBM = equity_models.GeometricBrownianMotion
        self.gbm = GBM(mu=0, sigma=0.2, s_zero=100, delta_t_in=1, delta_t_out=1)
        self.ppl = pipeline.Pipeline([self.rng, self.gbm])
        
    def test_returns_object(self):
        r_nb = self.ppl.generate()
        self.assertEqual(type(r_nb), np.ndarray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (10,5),
                         'incorrect shape')


class pipeline_gbm_test_xr(unittest.TestCase):
    """Test pipeline uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.NormalRng(shape={'svar':2, 'sim':10, 'time':5},
                                mean=[0,0], cov=[[1,0],[0,1]])
        GBM = equity_models.GeometricBrownianMotion
        self.gbm = GBM(mu=0, sigma=0.2, s_zero=100, delta_t_in=1, delta_t_out=1)

        self.ppl = pipeline.Pipeline([self.rng, self.gbm])
        
    def test_returns_object(self):
        r_nb = self.ppl.generate()
        self.assertEqual(type(r_nb), xr.DataArray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (2, 10,5))

    
        
if __name__ == '__main__':
    unittest.main()              
            