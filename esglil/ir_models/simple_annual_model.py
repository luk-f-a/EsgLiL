#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a simple annual model, based purely on a time-discrete logic. It
does not have an underlying continuous model.

Created on Sun May 13 13:57:42 2018

@author: luk-f-a
"""

from esglil.common import StochasticVariable

class ShortRate(StochasticVariable):
    """class for a simple short rate model
    This class only implements the short rate
    SDE: r(t+1)= ar(t)+b(t)+sigma*N(i+1)
         
     
     Parameters
    ----------
        
    b(t) : scalar or SDE object
        
    a: scalar or SDE object
        
    sigma : scalar
        standard deviation
        
    r_zero: scalar
        initial value of the short rate
        
    N: stochastic variable or random number generator
        standar normal variable
        
        
    """    
    __slots__ = ('b', 'a', 'sigma', 'N')
    
    def __init__(self, b, a, sigma, r_zero, N):
        self.b = b
        self.a = a
        self.sigma = sigma
        self.N = N
        self.t_1 = 0
        self.value_t = r_zero #.value_t
        #self._check_valid_params()

    def run_step(self, t):
        assert t-self.t_1 == 1
        self.value_t = self.a*self.value_t+self.b+self.sigma*self.N
        self.t_1 = t

    def run_step_ne(self, t):
        self._evaluate_ne('a*self_1+b+sigma*N)',  out_var='value_t')
        self.t_1 = t    
        
class CashAccount(StochasticVariable):
    """class for the cash account under the simple annual model

   
     Parameters
    ----------
    r: stochastic varible
        short rate
    """    
    __slots__ = ('r')
    
    def __init__(self, r):
        self.r = r
        self.value_t = 1
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.value_t*np.exp(self.r*1)
        self.t_1 = t

    def run_step_ne(self, t):
        self._evaluate_ne('self_1*exp(r)',   out_var='value_t')
        self.t_1 = t

