#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:49:10 2017

@author: luk-f-a
"""

import unittest
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from esglil.common import Variable, VariableView, TimeDependentParameter

import numpy as np
import pandas as pd
import operator

#TODO: test delta_t_in and out different than one and different than each other

        
class test_variable_math(unittest.TestCase):
    def test(self):
        var1 = Variable()
        var1.value_t = np.arange(1, 11)
        var2 = Variable()
        var2.value_t = np.arange(1,11)
        for op in [operator.add, operator.sub, operator.truediv,
                   operator.mul, operator.pow]:
            with self.subTest(op=str(op)):
                self.assertTrue(np.array_equal(op(var1, 10), 
                                               op(var1.value_t, 10)))
                self.assertTrue(np.array_equal(op(10, var1), 
                                               op(10, var1.value_t)))

        for op in [operator.add, operator.sub, operator.truediv,
                   operator.mul, operator.matmul]:
            with self.subTest(op=str(op)):
                self.assertTrue(np.array_equal(op(var2, var1), 
                                            op(var1.value_t, var2.value_t)))
                self.assertTrue(np.array_equal(op(var2, var1), 
                                               op(var2.value_t, var1.value_t)))

    def test_matmul(self):
        ## vectors
        var1 = Variable()
        var1.value_t = np.arange(1, 11)
        arr = np.arange(1,11)
        self.assertEqual(var1 @ arr, arr@arr)
        self.assertEqual(var1@arr, arr@var1)

        ## arrays
        var1 = Variable()
        var1.value_t = np.arange(1, 51).reshape(5, 10)
        arr = np.arange(1, 51).reshape(5, 10)
        self.assertTrue(np.array_equal(var1 @ arr.T, arr@arr.T))
        self.assertTrue(np.array_equal(arr.T @ var1, arr.T @ arr))

class test_variable_view_math(unittest.TestCase):
    def test(self):
        var1 = Variable()
        var1.value_t = np.arange(1, 21).reshape(2,10)
        var2 = Variable()
        var2.value_t = np.arange(1,11)
        var3 = Variable()
        var3.value_t = np.arange(51,61)
        for op in [operator.add, operator.sub, operator.truediv,
                   operator.mul, operator.pow]:
            with self.subTest(op=str(op)):
                self.assertTrue(np.array_equal(op(var1[0], 10), 
                                               op(var2, 10)))
                self.assertTrue(np.array_equal(op(10, var1[0]), 
                                               op(10, var2)))

        for op in [operator.add, operator.sub, operator.truediv,
                   operator.mul, operator.matmul]:
            with self.subTest(op=str(op)):
                self.assertTrue(np.array_equal(op(var3, var1[0]), 
                                            op(var3, var2)))
                self.assertTrue(np.array_equal(op(var1[0], var3), 
                                            op(var2, var3)))

        #Test that references do not remain attached to the old object
        var1.value_t = np.arange(11, 31).reshape(2,10)
        var2.value_t = np.arange(11, 21)
        for op in [operator.add, operator.sub, operator.truediv,
                   operator.mul, operator.pow]:
            with self.subTest(op=str(op)):
                self.assertTrue(np.array_equal(op(var1[0], 10), 
                                               op(var2, 10)))
                self.assertTrue(np.array_equal(op(10, var1[0]), 
                                               op(10, var2)))

    def test_matmul(self):
        ## vectors
        var1 = Variable()
        var1.value_t = np.stack([np.arange(1, 11), np.arange(1, 11)], axis=0)
        arr = np.arange(1,11)
        self.assertEqual(var1[0] @ arr, arr.T@arr)
        self.assertEqual(var1[1] @ arr, arr @ arr)
        self.assertEqual(var1[0]@arr, arr@var1[0])
        self.assertEqual(var1[1] @ arr, arr @ var1[1])

        ## arrays
        var1 = Variable()
        var1.value_t = np.stack([np.arange(1, 51).reshape(5, 10),
                                       np.arange(1, 51).reshape(5, 10)],
                                      axis=0)
        arr = np.arange(1, 51).reshape(5, 10)
        self.assertTrue(np.array_equal(var1[0] @ arr.T, arr@arr.T))
        self.assertTrue(np.array_equal(arr.T @ var1[0], arr.T @ arr))
        self.assertTrue(np.array_equal(var1[1] @ arr.T, arr@arr.T))
        self.assertTrue(np.array_equal(arr.T @ var1[1], arr.T @ arr))

if __name__ == '__main__':
    unittest.main()    