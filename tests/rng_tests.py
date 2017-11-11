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
import numpy as np
import xarray as xr


class Uniform_Rng_test_np(unittest.TestCase):
    """Test uniform rng with numpy output
    """
    def setUp(self):
        self.rng = rng.UniformRng(shape=(10,5))

    def test_returns_object(self):
        r_nb = self.rng.generate()
        self.assertEqual(type(r_nb), np.ndarray,
                         'incorrect type')
        self.assertEqual(r_nb.shape, (10,5),
                         'incorrect shape')


    def test_loop(self):
        for i, r_nb in enumerate(self.rng):
            if i == 5:
                break
            self.assertEqual(r_nb.shape, (10,5),
                             'incorrect shape')
            
class Uniform_Rng_test_xr_noloop(unittest.TestCase):
    """Test uniform rng with xarray output
    """
    def setUp(self):
        self.rng = rng.UniformRng(shape={'sim': 10, 
                            'svar': 5,
                            'time': 1})

    def test_returns_object(self):
        r_nb = self.rng.generate()
        self.assertEqual(type(r_nb), xr.DataArray, 'incorrect type')
        self.assertEqual(r_nb.shape, (5,10,1), 'incorrect shape')
        self.assertEqual(r_nb.dims, ('svar', 'sim', 'time'), 'incorrect dims')
        self.assertEqual(r_nb.coords['time'], 1, 'incorrect type coord')

    def test_loop(self):
        for i, r_nb in enumerate(self.rng):
            if i == 5:
                break
            self.assertEqual(r_nb.shape, (5,10,1), 'incorrect shape')
        self.assertEqual(r_nb.dims, ('svar', 'sim', 'time'), 'incorrect dims')
        self.assertEqual(r_nb.coords['time'], 1, 'incorrect type coord')

class Uniform_Rng_test_xr_withloop(unittest.TestCase):
    """Test uniform rng with xarray output and loop
    """
    def setUp(self):
        self.rng = rng.UniformRng(shape={'sim': 10, 
                            'svar': 5,
                            'time': 1}, loop_dim='time')

    def test_returns_object(self):
        r_nb = self.rng.generate()
        self.assertEqual(type(r_nb), xr.DataArray, 'incorrect type')
        self.assertEqual(r_nb.shape, (5,10,1), 'incorrect shape')
        self.assertEqual(r_nb.dims, ('svar', 'sim', 'time'), 'incorrect dims')
        self.assertEqual(r_nb.coords['time'], 1, 'incorrect type coord')

    def test_loop(self):
        for i, r_nb in enumerate(self.rng):
            if i == 5:
                break
            self.assertEqual(r_nb.shape, (5,10,1), 'incorrect shape')
        self.assertEqual(r_nb.dims, ('svar', 'sim', 'time'), 'incorrect dims')
        self.assertEqual(r_nb.coords['time'], i+1, 'incorrect type coord')   


class Uniform_Rng_test_xr_withloop2(unittest.TestCase):
    """Test uniform rng with xarray output and loop, bigger temp size
    """
    def setUp(self):
        self.rng = rng.UniformRng(shape={'sim': 10, 
                            'svar': 5,
                            'time': 2}, loop_dim='time')

    def test_returns_object(self):
        r_nb = self.rng.generate()
        self.assertEqual(type(r_nb), xr.DataArray, 'incorrect type')
        self.assertEqual(r_nb.shape, (5,10,2), 'incorrect shape')
        self.assertEqual(r_nb.dims, ('svar', 'sim', 'time'), 'incorrect dims')
        self.assertEqual(r_nb.coords['time'].values.tolist(), [1 ,2],
                         'incorrect type coord')

    def test_loop(self):
        for i, r_nb in enumerate(self.rng):
            if i == 5:
                break
            self.assertEqual(r_nb.shape, (5,10,2), 'incorrect shape')
        self.assertEqual(r_nb.dims, ('svar', 'sim', 'time'), 'incorrect dims')
        self.assertEqual(r_nb.coords['time'].values.tolist(), [2*i+1, 2*i+2],
                         'incorrect type coord') 
         
if __name__ == '__main__':
    unittest.main()              
            