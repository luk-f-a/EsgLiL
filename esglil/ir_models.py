#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:53:54 2017

@author: luk-f-a
"""
import xarray as xr

class HullWhite1FModel(object):
    """class for (1 Factor) Hull White model of short interest rate
    SDE: dr(t)=[b(t)-a*r(t)]dt+sigma(t)dW(t)
    
     Parameters
    ----------
    b(t) : DataArray
        mean reversion level
        
    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation
        
    r_zero : scalar
        initial value of short rate.
        
    delta_t : scalar
        
    """
    
    __slots__ = ('b', 'a', 'sigma', 'delta_t_in', 'delta_t_out', 'r', '_use_xr')
    
    def __init__(self, b=None, a=None, sigma=None, r_zero=1, delta_t_out=1, 
                 delta_t_in=1):
        self.b = b
        self.a = a
        self.sigma = sigma
        self.delta_t_in = delta_t_in
        self.delta_t_out = delta_t_out
        self.r = r_zero
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
        Produces simulations of 1 factor Hull White model short rate
        
        Parameters
        ----------       
        X array of Brownian motion increments (dW) simulations
        
        
        Returns
        -------
        GBM simulations calculated as S(t)=S(t-1)+ΔS(t)
                                        ΔS(t)=μS(t-1)dt+σS(t-1)X
        
        """
        assert not (self.a is None or self.b is None or self.sigma is None)
        a = self.a
        if type(self.b) is xr.DataArray:
            b = lambda t: self.b.loc[{'time':t}]
        elif callable(self.b):
            b = self.b
        else:
            raise TypeError
            
        sigma = self.sigma 
        self._check_valid_params()
        self._check_valid_X(X)
        dt_ratio = int(self.delta_t_out/self.delta_t_in)
        if self._use_xr:
            r = X.copy()
            r_t = self.r
            for t in r.time:
                b_t = b(t)
                r_t += (b_t-a*r_t)*self.delta_t_in+sigma*X.loc[{'time':t}]
                r.loc[{'time':t}] = r_t
            self.r = r_t
            r_out = r.loc[{'time':slice(dt_ratio,None,dt_ratio)}]
        else:
            self.r += (b_t-a*r_t)*self.delta_t_in+sigma*X.loc[{'time':t}]
            r_out = self.r
        return r_out