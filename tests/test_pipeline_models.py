#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:27:26 2017

@author: luk-f-a
"""
import unittest
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import pipeline_models

import numpy as np
import xarray as xr



class gbm_test(unittest.TestCase):
        
#    def test_shape_noloop2(self):
#        x = pipeline_models.GBM2(sims=20, loop_time=False)
#       
#        
#        self.assertEqual(type(x), xr.DataArray,
#                         'incorrect type')
#        self.assertEqual(x.shape, (20,10))
        
    def test_shape_noloop3a(self):
        gbm = pipeline_models.GBM3(sims=20, loop_time=False)
        x = gbm.generate()
        
        self.assertEqual(type(x), xr.DataArray,
                         'incorrect type')
        self.assertEqual(x.shape, (20,10))

    def test_shape_noloop3b(self):
        gbm = pipeline_models.GBM3(sims=20, time_sampling_ratio=10, 
                                   loop_time=False)
        x = gbm.generate()
        
        self.assertEqual(type(x), xr.DataArray,
                         'incorrect type')
        self.assertEqual(x.shape, (20,10))

#    def test_shape_loop2(self):
#        gbm = pipeline_models.GBM2(sims=20, loop_time=True)
#       
#        for x in gbm:
#            self.assertEqual(type(x), xr.DataArray,
#                             'incorrect type')
#            self.assertEqual(x.shape, (20,10))
    
    def test_shape_loop3(self):
        gbm = pipeline_models.GBM3(sims=20, loop_time=True)
        for x in gbm.time_loop_gen():
            self.assertEqual(type(x), xr.DataArray,
                             'incorrect type')
            self.assertEqual(x.shape, (20,10))

    def test_saving_variables_noloop(self):
        gbm = pipeline_models.GBM3(sims=20, loop_time=False, record_bm=True)
        y = gbm.generate()
        x = gbm.get_basic_rv()
        self.assertEqual(type(x), xr.DataArray,
                         'incorrect type')
        self.assertEqual(x.shape, (20,10))
        
    def test_saving_variables_loop(self):
        gbm = pipeline_models.GBM3(sims=20, loop_time=True, record_bm=True)

        
        for i, x in enumerate(gbm.time_loop_gen()):
            rv = gbm.get_basic_rv()
            self.assertEqual(type(rv), xr.DataArray,
                             'incorrect type')
            self.assertEqual(rv.shape, (20,10*(1+i)))
            
if __name__ == '__main__':
    unittest.main()  