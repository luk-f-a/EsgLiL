#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:28:07 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import FunctionOfVariable
try:
    import numexpr as ne
except ModuleNotFoundError:
    pass
except:
    raise

class GeometricBrownianMotion(FunctionOfVariable):
    """class for Geometric Brownian Motion model of equity returns with an
    Euler scheme
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

    def run_step_ne(self, t):
        self._evaluate_ne('self_1*(1 + mu*(t-t_1)+dW*sigma)', 
                          local_vars={'t': t}, out_var='value_t')
        self.t_1 = t
        
        
class GBM_exact(FunctionOfVariable):
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
    __slots__ = ('mu', 'sigma', 'W', 's_zero')   
    
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
        return self
    
    def run_step_ne(self, t):
        self._evaluate_ne('s_zero*exp(mu-0.5*sigma**2)*t+W*sigma', 
                          local_vars={'t': t}, out_var='value_t')

class EquityExcessReturns(FunctionOfVariable):
    """class for Geometric Brownian Motion model of equity returns
        calculated from the solution to the SDE (dS/S = mu*dt + sigma*dW)
        as S = S_0*exp((mu-0.5*sigma**2/2)*t+sigma*W) instead of using
        Euler on the SDE
        without drift and applied on top of a cash index
        
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
    __slots__ = ('cash', 'sigma', 'Z',  'exc_t', 's_zero')   
    
    def __init__(self, cash, sigma, Z, s_zero=100):
        self.cash = cash
        self.sigma = sigma
        self.Z = Z
        self.t_1 = 0
        self.s_zero = s_zero
        self.value_t = s_zero
        self.exc_t = 1
        #self._check_valid_params()

    def run_step(self, t):
        self.exc_t = self.exc_t*np.exp((-0.5*self.sigma**2)+self.Z*self.sigma)
        self.value_t = self.cash*self.s_zero*self.exc_t
        return self
    
  

class EquitySimpleAnnualModel(FunctionOfVariable):
    """class for a simple annual model of equity returns
        
     Parameters
    ----------
     sigma : scalar or SDE
        standard deviation of gbm
        
    s_zero : scalar or SDE
        initial value of gbm. Defaults to 100.
        
    N: stochastic variable or random number generator
        standar normal variable    
       
    """
    __slots__ = ('r', 'sigma', 'N')   
    
    def __init__(self, r, sigma, N, s_zero=100):
        self.r = r
        self.sigma = sigma
        self.N = N
        self.t_1 = 0
        self.value_t = s_zero
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.value_t*np.exp((self.r-0.5*self.sigma**2)+self.N*self.sigma)
        return self
    
    def run_step_ne(self, t):
        self._evaluate_ne('self_1*exp(r-0.5*sigma**2)*t+N*sigma', 
                          out_var='value_t')    