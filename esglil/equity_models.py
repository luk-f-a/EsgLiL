#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:28:07 2017

@author: lucio
"""

class GeometricBrownianMotion(object):
    """class for Geometric Brownian Motion model of equity returns

     Parameters
    ----------
    """
    __slots__ = ('mu', 'sigma')
    
    def __init__(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        
    def _check_valid_params():
        pass
        
    def generate(self, X):
        """produces simulations of  Geometric Brownian Motion
        
        Parameters
        ----------       
        X array of Brownian motion simulations
        """
        assert not (self.mu is None or self.sigma is None)
        self._check_valid_params()
        return np.exp(mu)
        
        
        
    