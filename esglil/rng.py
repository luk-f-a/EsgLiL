#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:27:12 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import Variable
from esglil.common import SDE
from esglil.sobol import i4_sobol_std_normal_generator as sobol_normal
from esglil.multithreaded_rng import MultithreadedRNG
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

 
class IndWienerIncr(Rng):
    """class for independent increments of Wiener process

     Parameters
    ----------
    dims: int
        Amount of dimensions in for the output normal variable
    
    sims : int
        Amount of simulations to produce in each timestep
        
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
        
    delta_t : float or fraction
        Time step of the increment from which the variance is derived
        
    generator: string {'mc-numpy', ''mc-dask'}
        Type of generator. 
        mc-numpy: Monte Carlo numpy generator
        mc-dask: Monte Carlo generator that uses dask for parallel calculations
        
    dask_chunks: int
        In how many parts should the simulations be split for dask
        In a Windows machine (even though it had 2 cores), 1 was the best choice
        
        
        
    """
    __slots__ = ('mean', 'delta_t', 'generator')
                 
    def __init__(self,  dims, sims, mean=0, delta_t=1, generator='mc-numpy',
                 dask_chunks=1):
        Rng.__init__(self, dims, sims)
        
        assert generator in ('mc-numpy', 'mc-dask', 'sobol-np')
        self.mean = mean
        self.delta_t = delta_t
        if generator == 'mc-dask':
            import dask.array as da
            self.generator = lambda :da.random.normal(mean, np.sqrt(delta_t), 
                             size=(dims, sims), chunks=int(sims/dask_chunks))
        elif generator == 'mc-numpy':
            self.generator = lambda :np.random.normal(mean, np.sqrt(delta_t), 
                             size=(dims, sims))
        elif generator == 'sobol-np':
            s_gen = sobol_normal(dims, sims)
            self.generator = lambda : (mean+np.sqrt(delta_t)*next(s_gen)).T

      
        
    def _check_valid_params(self):
        #TODO: if output is numpy, mean must be size 1 and cov 1x1
        # if output is xr size of mean and cov must agree with svar dim
        pass
    
    
    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = self.generator()
#        print(out.squeeze()[:2])
        return out.squeeze()

class MCMVIndWienerIncr_old(Rng):
    """class for class for independent increments of Wiener process with 
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
        
    delta_t : float or fraction
        Time step of the increment from which the variance is derived
        
    generator: string {'mc-numpy', ''mc-dask'}
        Type of generator. 
        mc-numpy: Monte Carlo numpy generator
        mc-dask: Monte Carlo generator that uses dask for parallel calculations
        
    dask_chunks: int
        In how many parts should the simulations be split for dask
        In a Windows machine (even though it had 2 cores), 1 was the best choice
         
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
    __slots__ = ('mean', 'delta_t', 'sims_inner', 'sims_outer', 'mcmv_time',
                 'fixed_arrival', 'generator', 'library', 'chunks')
                 
    def __init__(self,  dims, sims_outer, sims_inner, mean, delta_t, 
                 mcmv_time, fixed_inner_arrival=True, generator='mc-numpy',
                 dask_chunks=1):
        Rng.__init__(self, dims, sims_outer*sims_inner)
        self.mean = mean
        self.sims_inner = sims_inner
        self.sims_outer = sims_outer
        self.mcmv_time = mcmv_time
        self.fixed_arrival = fixed_inner_arrival
        if generator == 'mc-dask':
            import dask.array as da
            self.library = da
            chunks = int((sims_outer * sims_inner)/dask_chunks)
            self.chunks = chunks
            self.generator = lambda sims:da.random.normal(mean, np.sqrt(delta_t), 
                             size=(dims, sims), chunks=chunks)
        elif generator == 'mc-numpy':
            self.library = np
            self.generator = lambda sims:np.random.normal(mean, np.sqrt(delta_t), 
                             size=(dims, sims))        
    def _check_valid_params(self):
        #TODO: if output is numpy, mean must be size 1 and cov 1x1
        # if output is xr size of mean and cov must agree with svar dim
        pass
    
    def run_step(self, t):
        #self.value_t[...] = self.generate()
        if t <= self.mcmv_time:
            rn = self.generate(self.sims_outer)
#            self.value_t  = self.library.tile(rn, [1,self.sims_inner]).squeeze()
            rn  = self.library.tile(rn, self.sims_inner).squeeze()
            if hasattr(rn, 'rechunk'):
                rn = rn.rechunk(self.chunks)
            self.value_t = rn
        else:
            self.value_t  = self.generate(self.sims_outer*self.sims_inner)
            
    def generate(self, sims):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = self.generator(sims)
#        print(out.squeeze()[:2])
        return out.squeeze()
        
    
class MCMVIndWienerIncr(Rng):
    """class for class for independent increments of Wiener process with 
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
        
    delta_t : float or fraction
        Time step of the increment from which the variance is derived
        
    generator: string {'mc-numpy', ''mc-dask'}
        Type of generator. 
        mc-numpy: Monte Carlo numpy generator
        mc-dask: Monte Carlo generator that uses dask for parallel calculations
        
    dask_chunks: int
        In how many parts should the simulations be split for dask
        In a Windows machine (even though it had 2 cores), 1 was the best choice
         
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
    __slots__ = ('mean', 'delta_t', 'sims_inner', 'sims_outer', 'mcmv_time',
                 'fixed_arrival', 'generator', 'dask_generator', 'chunks',
                 'multithr_generator')
                 
    def __init__(self,  dims, sims_outer, sims_inner, mean, delta_t, 
                 mcmv_time, fixed_inner_arrival=True, generator='mc-numpy',
                 n_jobs=1):
        Rng.__init__(self, dims, sims_outer*sims_inner)
        self.mean = mean
        self.sims_inner = sims_inner
        self.sims_outer = sims_outer
        self.mcmv_time = mcmv_time
        self.fixed_arrival = fixed_inner_arrival

        self.generator = lambda sims:np.random.normal(mean, np.sqrt(delta_t), 
                         size=(dims, sims))   
        self.dask_generator = None
        self.multithr_generator = None
        if generator == 'mc-dask':
            import dask.array as da
            chunks = int((sims_outer * sims_inner)/n_jobs)
            self.chunks = chunks
            self.dask_generator = lambda sims:da.random.normal(mean, np.sqrt(delta_t), 
                             size=(dims, sims), chunks=chunks)
        if generator == 'mc-multithreaded':
            self.dask_generator = None
            rgen = MultithreadedRNG(dims*sims_outer * sims_inner, threads=n_jobs)
            std = np.sqrt(delta_t)
            self.multithr_generator = lambda sims: (mean+std*rgen.fill().values).reshape((dims, sims))


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
            if self.dask_generator is not None:
                self.generator = self.dask_generator
            if self.multithr_generator is not None:
                self.generator = self.multithr_generator
            self.value_t  = self.generate(self.sims_outer*self.sims_inner)
            
    def generate(self, sims):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = self.generator(sims)
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
                 mcmv_time, fixed_inner_arrival=True, generator='mc-numpy',
                 dask_chunks=1):
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