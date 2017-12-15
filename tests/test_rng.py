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
        df_full_run = self.esg.full_run_to_pandas(dt_out=1, max_t=40)
        self.assertEqual(type(df_full_run), pd.DataFrame,
                         'incorrect type')
        self.assertEqual(df_full_run.shape, (10*40, 2))
         
if __name__ == '__main__':
    unittest.main()              
            