#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:27:12 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import Variable
from esglil.common import SDE
from collections import Iterable

class Rng(Variable):
    """Base class for random number generators
    """
    __slots__ = ('shape', 'value_t', 'sims', 'dims')

    def __init__(self, dims, sims):
        self.sims = sims
        self.dims = dims
        if dims==1:
            self.shape = (sims,)
        else:
            self.shape = (dims, sims)
#        if isinstance(sims, Iterable):
#            self.shape = [dims]+[s for s in sims]
#        else:
#            self.shape = [dims, sims]
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

 
class IndependentWienerIncrements(Rng):
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
    __slots__ = ('mean', 'delta_t', 'library')
                 
    def __init__(self,  dims, sims, mean=0, delta_t=1, distributed=False):
        Rng.__init__(self, dims, sims)
        self.mean = mean
        self.delta_t = delta_t
        if distributed:
            import dask.array as da
            self.library = da
        else:
            self.library = np
        
    def _check_valid_params(self):
        #TODO: if output is numpy, mean must be size 1 and cov 1x1
        # if output is xr size of mean and cov must agree with svar dim
        pass
    
    
    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        kwargs = {}
        if not self.library is np:
            kwargs['chunks'] = int(self.sims/4)
        out = self.library.random.normal(self.mean, 
                                         self.delta_t, 
                                         size=(self.dims, self.sims), **kwargs )
#        print(out.squeeze()[:2])
        return out.squeeze()


    
class MCMVNormalRng(Rng):
    """class for normal random number generation with 
    special structure for MonteCarlo pricing at t>0

     Parameters
    ----------
    dims: int
        Amount of dimensions in for the output normal variable
    
    sims_outer : int
        Amount of simulations to produce in each timestep for the
        outer simulation
        
    sims_inner : int
        Amount of simulations to produce in each timestep for the
        inner simulation
        
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
        
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. 
        It must be symmetric and positive-semidefinite for proper sampling.
        
    mcmv_time: scalar
        Time at which the MC valuation will be performed. At this time
        there will be sims_inner*sims_outer total simulations but only 
        simg_outer of them will be different values
        
    fixed_inner_arrival: True or False
        If true, for each of the sims_outer different values at time t, the
        arrival path (simulations before t) will be identical.
        If false, each of the identical sims_inner at time t, will not have
        identical values at time<t, but randomized to arrive via 
        different paths using a brownian bridge. This is only appropriate
        for functions which are not path dependent, only dependent on t. This 
        second option is not yet implemented
    """
    __slots__ = ('mean', 'cov', 'sims_inner', 'sims_outer', 'mcmv_time',
                 'fixed_arrival')
                 
    def __init__(self,  dims, sims_outer, sims_inner, mean, cov, 
                 mcmv_time, fixed_inner_arrival=True):
        Rng.__init__(self, dims, sims_outer*sims_inner)
        self.mean = mean
        self.cov = cov
        self.sims_inner = sims_inner
        self.sims_outer = sims_outer
        self.mcmv_time = mcmv_time
        self.fixed_arrival = fixed_inner_arrival
        
    def _check_valid_params(self):
        #TODO: if output is numpy, mean must be size 1 and cov 1x1
        # if output is xr size of mean and cov must agree with svar dim
        pass
    
    def run_step(self, t):
        #self.value_t[...] = self.generate()
        if t <= self.mcmv_time:
            rn = self.generate(self.sims_outer)
            self.value_t  = np.tile(rn, [1,self.sims_inner]).squeeze()
        else:
            self.value_t  = self.generate(self.sims_outer*self.sims_inner)
            
    def generate(self, sims):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = np.random.multivariate_normal(self.mean, self.cov, 
                                               size=sims,
                                               check_valid='raise').T
#        print(out.squeeze()[:2])
        return out.squeeze()
    
    
class WienerProcess(SDE):
    """class for accumulating Wiener increments into a running sum
  
     Parameters
    ----------

    dW: random number generator of Wiener increments
    """    
    __slots__ = ['dW']
    
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
    __slots__ = ('rng', 'm')
                 
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
        if hasattr(self.rng(), 'compute'):
            import dask.array as da
            self.value_t = da.from_array(self.m, chunks=1000)@self.rng()            
        else:
            self.value_t = self.m@self.rng()
        
    
class PreCalculatedFeed(SDE):
    """class for feeding a precalculated stream (usually quadrature points 
    for dW)
    
    points: 3-d array
    """
 
    __slots__ = ('points', 'col')
                 
    def __init__(self,  points):
        self.points = points
        self.col = 0
        
    def run_step(self, t):    
        self.value_t = self.points[self.col, :,:]
        self.col += 1