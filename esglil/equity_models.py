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
    __slots__ = ('mu', 'sigma', 'dW')   
    
    def __init__(self, mu, sigma, dW, s_zero=100):
        self.mu = mu
        self.sigma = sigma
        self.dW = dW
        self.t_1 = 0
        self.value_t = s_zero
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.value_t*(1 + self.mu*(t-self.t_1)+self.dW*self.sigma)
        self.t_1 = t
        
        
        
class GBM_exact(SDE):
    """class for Geometric Brownian Motion model of equity returns
        calculated from the solution to the SDE (dS/S = mu*dt + sigma*dW)
        as S = S_0*exp((mu-0.5*sigma**2/2)*t+sigma*W) instead of using
        Euler on the SDE
        
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
    __slots__ = ('mu', 'sigma', 'W')   
    
    def __init__(self, mu, sigma, W, s_zero=100):
        self.mu = mu
        self.sigma = sigma
        self.W = W
        self.t_1 = 0
        self.s_zero = s_zero
        self.value_t = s_zero
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.s_zero*np.exp((self.mu-0.5*self.sigma**2)*t+self.W*self.sigma)
         