#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:53:54 2017

@author: luk-f-a
"""
import xarray as xr
import numpy as np

#class HullWhite1fZCBPrice(object):
#    """class for bond prices from (1 Factor) Hull White model 
#        P(t, T) = A(t, T)exp(−B(t,T)r(t))
#            B(t, T) = 1/a(1 − exp(−a(T −t))
#            A(t, T) = P(0, T)/P M(0, t)exp(B(t, T)f(0, t) −σ^2/(4a)(1 − exp(−2at))B(t, T)^2))
#            f(0, t) is the market instantaneous forward rate at time 0 for
#                    maturity T
#     Parameters
#    ----------
#    T: scalar, numpy array or DataArray
#        maturity of bond or bonds to be priced.
#        
#    P(t): DataArray or callable (with t as only argument)
#            ZCB price at time 0 for maturity t
#        
#    f(t) : DataArray or callable (with t as only argument)
#        market instantaneous forward rate at time 0 for maturity t
#        
#    a: scalar
#        mean reversion speed of hw model
#    
#    sigma : scalar
#        standard deviation
#        
#    delta_t : scalar
#        
#    """    
#
#    __slots__ = ('T', 'P', 'sigma', 'a', 'delta_t_in', 'delta_t_out', '_use_xr')
#    
#    def __init__(self, T=None, P=None, sigma=None, a=None, delta_t_out=1, 
#                 delta_t_in=1):
#        self.T = T
#        self.P = P
#        self.sigma = sigma
#        self.delta_t_in = delta_t_in
#        self.delta_t_out = delta_t_out
#        self._use_xr = None
#
#    def _check_valid_params(self):
#        assert self.delta_t_out >= self.delta_t_in
#        assert self.delta_t_out % self.delta_t_in == 0, ('delta out is {} and'
#        ' delta in is {}').format(self.delta_t_out, self.delta_t_in)
#
#    def _check_valid_X(self, X):
#        #TODO: check that X is either an xarray with the right dimensions
#        # or a numpy array with the right dimensions (time dimension should be of size 1)
#        if self._use_xr is None:
#            if type(X) is xr.DataArray:
#                self._use_xr = True
#            else:
#                self._use_xr = False
#        else:
#            if (type(X) is xr.DataArray) != (self._use_xr):
#                raise TypeError("Generate called with and without xarray input")
#        pass
#
#    def transform(self, X):
#        """
#        Produces simulations of 1 factor Hull White model short rate
#        
#        Parameters
#        ----------       
#        X array of hw 1f short rate simulations
#        
#        
#        Returns
#        -------
#        Zero coupon bond prices simulations for HW 1F model
#        
#        """
#        assert not (self.a is None or self.b is None or self.sigma is None)
#        a = self.a
#        if type(self.b) is xr.DataArray:
#            b = lambda t: self.b.loc[{'timestep':t}]
#        elif callable(self.b):
#            b = self.b
#        else:
#            raise TypeError
#            
#        sigma = self.sigma 
#        self._check_valid_params()
#        self._check_valid_X(X)
#        dt_ratio = int(self.delta_t_out/self.delta_t_in)
#        if not self._use_xr:
#            raise NotImplemented
#        p_t = X.copy()
#        r_t = self.r
#        for ts in r.timestep:
#            t = ts*self.delta_t_in
#            b_t = b(t)
#            r_t += (b_t-a*r_t)*self.delta_t_in+sigma*X.loc[{'timestep':t}]
#            r[{'timestep':t}] = r_t
#        self.r = r_t
#        r_out = r.loc[{'timestep':slice(dt_ratio, None, dt_ratio)}]
#            
#        return r_out

class HullWhite1fModel(object):
    """class for (1 Factor) Hull White model of short interest rate
    SDE: dr(t)=[b(t)-a*r(t)]dt+sigma(t)dW(t)
    
     Parameters
    ----------
    b(t) : DataArray or callable (with t as only argument)
        mean reversion level
        
    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation
        
    r_zero : scalar
        initial value of short rate.
        
    bond_prices: dictionary or DataArray
        initial bond prices to be moved forward in time. If passing a 
        dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values
        
    delta_t_in, delta_t_out: scalars
        
    """
    
    __slots__ = ('b', 'a', 'sigma', 'p_t', 'delta_t_in', 'delta_t_out', 'r',
                 '_use_xr')
    
    def __init__(self, b=None, a=None, sigma=None, r_zero=1, bond_prices=None,
                 delta_t_out=1, delta_t_in=1):
        self.b = b
        self.a = a
        self.sigma = sigma
        self.delta_t_in = delta_t_in
        self.delta_t_out = delta_t_out
        self.r = r_zero
        self._use_xr = None
        self.p_t = bond_prices
        
    def _check_valid_params(self):
        assert self.delta_t_out >= self.delta_t_in
        dt_ratio = round(self.delta_t_out/self.delta_t_in,5)
        assert dt_ratio == int(dt_ratio), ('delta out is {} and'
        ' delta in is {}. Ratio is {}').format(self.delta_t_out, self.delta_t_in,
                      round(self.delta_t_out / self.delta_t_in,5))

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
    
    def _bond_dict_to_xr(self, dict_):
        return xr.DataArray(list(dict_.values()), dims=['maturity'],
                                     coords=[list(dict_.keys())])
        
    def transform(self, X):
        """
        Produces simulations of 1 factor Hull White model short rate
        
        Parameters
        ----------       
        X array of Brownian motion increments (dW) simulations
        
        
        Returns
        -------
        Short rate simulations calculated as r(t)=r(t-1)+Δr(t)
                                        Δr(t)=[b(t)-a*r(t)]dt+sigma*X
        If bond_prices were provided initially, then they are simulated as
         ΔB(t)=B(t)*[r(t)]dt+sigma/a*(1-exp(-a(T-t)))*X
         and provided as separate svars
                                        
        
        """
        assert not (self.a is None or self.b is None or self.sigma is None)
        a = self.a
        if type(self.b) is xr.DataArray:
            b = lambda t: self.b.loc[{'timestep':t}]
        elif callable(self.b):
            b = self.b
        else:
            raise TypeError
        if type(self.p_t) is dict:
            self.p_t = self._bond_dict_to_xr(self.p_t)
        sigma = self.sigma 
        self._check_valid_params()
        self._check_valid_X(X)
        dt_ratio = int(round(self.delta_t_out/self.delta_t_in,5))
        if self._use_xr:
            r = xr.DataArray(np.empty(X.shape), dims=X.dims, coords=X.coords)
            r.coords['svar']=['short_rate']
            if self.p_t is not None: 
                n_bonds = len(self.p_t)
                empty_p = np.repeat(np.empty(X.shape), n_bonds, axis=0)
                bond_coords = ['bond_{}'.format(float(i)) 
                              for i in self.p_t.coords['maturity']]
                coords = dict(X.coords)
                coords['svar'] = bond_coords
                p = xr.DataArray(empty_p, dims=X.dims, coords=coords)
                
            r_t = self.r
            p_t = self.p_t
            for ts in r.timestep:
                t = ts*self.delta_t_in
                b_t = b(t)
                r_t += (b_t-a*r_t)*self.delta_t_in+sigma*X.loc[{'timestep':ts}]
                r.loc[{'timestep':ts}] = r_t
                #if bond prices must be calcualted, then move them forward
                if self.p_t is not None: 
                    T = self.p_t.coords['maturity']
                    p_t = p_t + p_t*(r_t+sigma/a*(1-np.exp(-a*(T-t))))
            self.r = r_t
            self.p_t = p_t
            r_out = r[{'timestep':slice(dt_ratio-1, None, dt_ratio)}]
            if self.p_t is not None: 
                p_out = p[{'timestep':slice(dt_ratio-1, None, dt_ratio)}]
                out = xr.concat([r_out, p_out], dim='svar')
            else:
                out = r_out
            out.attrs['delta_t'] = self.delta_t_out
        else:
            self.r += (b_t-a*r_t)*self.delta_t_in+sigma*X.loc[{'timestep':t}]
            out = self.r
        return out