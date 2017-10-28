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
import numpy as np
import xarray as xr


class pipeline_uniform_rng_test_np(unittest.TestCase):
    """Test pipeline uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.UniformRng(shape=(10,5))
        self.ppl = pipeline.Pipeline([self.rng])
        
    def test_returns_object(self):
        r_nb = self.ppl.generate()
        self.assertEqual(type(r_nb), np.ndarray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (10,5),
                         'incorrect shape')


    def test_loop(self):
        for i, r_nb in enumerate(self.ppl):
            if i == 5:
                break
            self.assertEqual(r_nb.shape, (10,5),
                             'incorrect shape')

class model_union_uniform_rng_test_np(unittest.TestCase):
    """Test pipeline uniform rng with numpy output
    """
    def setUp(self):
        self.rng1 = rng.UniformRng({'svar':1, 'sim':10, 'time':5})
        self.rng2 = rng.UniformRng({'svar':1, 'sim':10, 'time':5})
        self.mu = pipeline.ModelUnion([('a', self.rng1), ('b', self.rng2)])
        
    def test_returns_object(self):
        r_nb = self.mu.generate()
        self.assertEqual(type(r_nb), xr.DataArray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (2, 10,5),
                         'incorrect shape')


    def test_loop(self):
        for i, r_nb in enumerate(self.mu):
            if i == 5:
                break
            self.assertEqual(r_nb.shape, (10,5),
                             'incorrect shape')      
        
if __name__ == '__main__':
    unittest.main()              
            