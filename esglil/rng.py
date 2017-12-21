#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:27:12 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import Variable
from esglil.common import SDE

class Rng(Variable):
    """Base class for random number generators
    """
    __slots__ = ('shape', 'value_t')

    def __init__(self, dims, sims):
        self.sims = sims
        #self.value_t = np.zeros(shape=(dims, sims))
                
    def run_step(self, t):
        #self.value_t[...] = self.generate()
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


class NormalRng(Rng):
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
                                               size=self.sims,
                                               check_valid='raise').T
#        print(out.squeeze()[:2])
        return out.squeeze()
    
class WienerProcess(SDE):
    """class for accumulating Wiener increments into a running sum
  
     Parameters
    ----------

    dW: random number generator of Wiener increments
    """    
    __slots__ = ('dW')
    
    def __init__(self, dW):
        self.value_t = 0
        self.dW = dW

    def run_step(self, t):
        self.value_t = self.dW + self.value_t
   
class CorrelatedRV(SDE):
    """class for correlating independent variables
  
     Parameters
    ----------

     cov: covariance matrix
     rng: random number generator for independent variables with variance = 1
    """
    __slots__ = ('rng', 'cov')
                 
    def __init__(self,  rng, input_cov, target_cov):
        """covar calculations
        target_cov = L.L'  (L is cholesky)
        X: independent dW
        Z: dependednt dW
        to obtain E[ZZ']= target_cov then
        Z=MX where M=L@[[sigma^-1, 0],[0, sigma^-1]]
        and sigma=sqrt(delta_t)
        """
        self.rng = rng
        l = np.linalg.cholesky(target_cov)
        inverse_sqrt_input_cov = np.diag(np.reciprocal(np.sqrt(np.diag(input_cov))))
        self.m = l@inverse_sqrt_input_cov
        
    def run_step(self, t):
        self.value_t = self.m@self.rng()

    
   