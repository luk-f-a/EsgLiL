#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:56:31 2018

@author: luk-f-a
"""
import numpy as np
from esglil.common import StochasticVariable

class DeterministicBankAccount(StochasticVariable):
    """for discounting under deterministic (constant) interest rates
    """
    __slots__ = ('r')
    
    def __init__(self, r):
        self.r = r
        self.value_t = 1
 
    def run_step(self, t):
        self.value_t = np.exp(self.r*t)