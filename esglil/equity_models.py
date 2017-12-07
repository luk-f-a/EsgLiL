#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:28:07 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import SDE
    
class GeometricBrownianMotion(SDE):
    """class for Geometric Brownian Motion model of equity returns
        dS/S = mu*dt + sigma*dW
     Parameters
    ----------
    mu : scalar or SDE
        drift
        
    sigma : scalar or SDE
        standard deviation of gbm
        
    s_zero : scalar or SDE
        initial value of gbm. Defaults to 1.
        
    delta_t : scalar
        
    """
    __slots__ = ('mu', 'sigma')   
    
    def __init__(self, mu, sigma, dW, s_zero=100):
        self.mu = mu
        self.sigma = sigma
        self.dW = dW
        self.t_1 = 0
        self.value_t = s_zero
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.value_t *(1 + self.mu*(t-self.t_1)+self.sigma*self.dW)
        self.t_1 = t
        
        
        
    