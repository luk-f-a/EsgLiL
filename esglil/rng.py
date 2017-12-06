#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:27:12 2017

@author: luk-f-a
"""


import numpy as np
from .common import Variable

class Rng(object):
    """Base class for random number generators
    """
    __slots__ = ('shape', 'value_t')

    def __init__(self, dims, sims):
        self.shape = (dims, sims)
                
    def run_step(self, t):
        self.value_t = self.generate()
        
    def generate(self):
        """Implement here the code to provide the next iteration of the
        random number generator
        """
        raise NotImplementedError
  


class UniformRng(Rng):
    """class for uniform random number generation

     Parameters
    ----------
    dims: int
        Amount of dimensions in for the output normal variable
    
    sims : int
        Amount of simulations to produce in each timestep

    """
    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = np.random.rand(*self.shape)
        return out


class NormalRng(Rng, Variable):
    """class for normal random number generation

     Parameters
    ----------
    dims: int
        Amount of dimensions in for the output normal variable
    
    sims : int
        Amount of simulations to produce in each timestep
        
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
        
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. 
        It must be symmetric and positive-semidefinite for proper sampling.
    """
    __slots__ = ('mean', 'cov')
                 
    def __init__(self,  dims, sims, mean, cov):
        Rng.__init__(self, dims, sims)
        self.mean = mean
        self.cov = cov
        
    def _check_valid_params(self):
        #TODO: if output is numpy, mean must be size 1 and cov 1x1
        # if output is xr size of mean and cov must agree with svar dim
        pass
    
    
    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        
        out = np.random.multivariate_normal(self.mean, self.cov, 
                                               self.shape,
                                               check_valid='raise')
        return out