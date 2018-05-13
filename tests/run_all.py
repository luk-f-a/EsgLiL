#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:51:00 2017

@author: luk-f-a
"""
import unittest
import sys, os


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.getcwd()))
#    from esglil import rng
#    from esglil import equity_models
    testsuite = unittest.TestLoader().discover('.')
    unittest.TextTestRunner(verbosity=1).run(testsuite)
