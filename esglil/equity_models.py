#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:28:07 2017

@author: luk-f-a
"""
import numpy as np
import xarray as xr

class GeometricBrownianMotion(object):
    """class for Geometric Brownian Motion model of equity returns

     Parameters
    ----------
    mu : scalar
        drift
        
    sigma : scalar
        standard deviation of gbm
        
    s_zero : scalar
        initial value of gbm. Defaults to 1.
        
    delta_t : scalar
        
    """
    __slots__ = ('mu', 'sigma', 'delta_t_in', 'delta_t_out', 'S', '_use_xr')
    
    def __init__(self, mu=None, sigma=None, s_zero=1, 
                 delta_t_out=1, delta_t_in=1):
        #TODO : handle non-scalar mu, sigma and s_zero
        self.mu = mu
        self.sigma = sigma
        self.delta_t_in = delta_t_in
        self.delta_t_out = delta_t_out
        self.S = s_zero
        self._use_xr = None
        
    def _check_valid_params(self):
        assert self.delta_t_out >= self.delta_t_in
        assert self.delta_t_out % self.delta_t_in == 0, ('delta out is {} and'
        ' delta in is {}').format(self.delta_t_out, self.delta_t_in)

    def _check_valid_X(self, X):
        #TODO: check that X is either an xarray with the right dimensions
        # or a numpy array with the right dimensions (time dimension should be of size 1)
        if self._use_xr is None:
            if type(X) is xr.DataArray:
                self._use_xr = True
            else:
                self._use_xr = False
        else:
            if (type(X) is xr.DataArray) != (self._use_xr):
                raise TypeError("Generate called with and without xarray input")
        pass
        
    def transform(self, X):
        """
        Produces simulations of Geometric Brownian Motion
        
        Parameters
        ----------       
        X array of Brownian motion increments (dW) simulations
        
        
        Returns
        -------
        GBM simulations calculated as S(t)=S(t-1)+ΔS(t)
                                        ΔS(t)=μS(t-1)dt+σS(t-1)X
        
        """
        assert not (self.mu is None or self.sigma is None)
        self._check_valid_params()
        self._check_valid_X(X)
        dt_ratio = int(self.delta_t_out/self.delta_t_in)
        if self._use_xr:
            S = X.copy()
            S_t = self.S
            for t in S.time:
                S_t = self.mu*S_t*self.delta_t_in+self.sigma*S_t*X.loc[{'time':t}]
                S.loc[{'time':t}] = S_t
            self.S = S_t
            S_out = S.loc[{'time':slice(dt_ratio,None,dt_ratio)}]
        else:
            self.S = self.mu*self.S*self.delta_t_in+self.sigma*self.S*X
            S_out = self.S
        return S_out
        
        
        
    