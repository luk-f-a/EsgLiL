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
#from matplotlib import pyplot as plt

#TODO: test delta_t_in and out different than one and different than each other

        
class hw1f_test_xr(unittest.TestCase):
    def setUp(self):
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':10, 'time':5},
                                mean=[0], cov=[[1]])
        HW = ir_models.HullWhite1FModel
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
#            x = np.insert(sims.coords['time'].values,0,0)
#            plt.plot(x, row)



class pipeline_hw1f_test_xr(unittest.TestCase):
    """Test pipeline uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.NormalRng(shape={'svar':1, 'sim':10, 'time':5},
                                mean=[0], cov=[[1]])
        HW = ir_models.HullWhite1FModel
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
            