#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:27:12 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import Variable
from esglil.common import StochasticVariable
from esglil.sobol import i4_sobol_std_normal_generator as sobol_normal
from esglil.multithreaded_rng import (MultithreadedRNG, BackgroundRNGenerator,
                                      BackgroundProcessRNGenerator)


class Rng(Variable):
    """Base class for random number generators
    """
    __slots__ = ('shape', 'sims', 'dims')

    def __init__(self, dims, sims):
        self.sims = sims
        self.dims = dims
        if dims == 1:
            self.shape = (sims,)
        else:
            self.shape = (dims, sims)

    def run_step(self, t):
        if t == 0:
            self.initialize()
        else:
            self.value_t = self.generate()

    def generate(self):
        """Implement here the code to provide the next iteration of the
        random number generator
        """
        raise NotImplementedError

    def initialize(self):
        self.value_t = np.zeros(shape=(self.dims, self.sims))

    def run_step_ne(self, t):
        self.run_step(t)



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
        out = np.random.rand(*self.shape)
        return out


class NormalRng(Rng):
    """class for multivariable normal random number generation with
    arbitrary covariance matrix.

    For the specific case of the generation of Wiener process increments
    please use IndWienerIncr class.

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

    def __init__(self, dims, sims, mean, cov):
        Rng.__init__(self, dims, sims)
        self.mean = mean
        self.cov = cov

    def _check_valid_params(self):
        pass


    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = np.random.multivariate_normal(self.mean, self.cov,
                                            size=self.sims,
                                            check_valid='raise').T
        return out

class PCAIndWienerIncr(Rng):
    __slots__ = ('mean', 'delta_t', 'generator', 'years', 'current_year',
                 'gen_rn')
                 
    def __init__(self,  dims, sims, mean=0, delta_t=1, years=1, generator='mc-numpy',
                 dask_chunks=1, seed=None, n_threads=1):
        Rng.__init__(self, dims, sims)
        self.mean = mean
        self.years = years
        self.delta_t = delta_t    
        self.gen_rn = np.random.normal(self.mean, np.sqrt(self.delta_t),
                             size=(self.dims*self.years, self.sims))
        from sklearn.decomposition import PCA
        pca = PCA(self.dims*self.years)
#        print(np.round(np.corrcoef(self.gen_rn, rowvar=True),2))
#        print('----')
#        print(np.round(np.corrcoef(pca.fit_transform(self.gen_rn.T), rowvar=False),2))
    
        self.gen_rn = pca.fit_transform(self.gen_rn.T).T.reshape(years, dims, sims)
#        print(np.round(np.corrcoef(self.gen_rn.reshape(years* dims, sims), rowvar=True),2))
    
        self.current_year = 0
        
    def generate(self):
        """Return the next iteration of the random number generator
        """
        out = self.gen_rn[self.current_year,:,:]
        self.current_year += 1
        return out
        
                
    
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
                 dask_chunks=1, seed=None, n_threads=1):
        Rng.__init__(self, dims, sims)

        assert generator in ('mc-numpy', 'mc-dask', 'sobol-np', 'mc-dask-fast',
                             'mc-multithreaded')
        self.mean = mean
        self.delta_t = delta_t
        
        if generator == 'mc-dask':
            import dask.array as da
            self.generator = lambda: da.random.normal(mean, np.sqrt(delta_t),
                             size=(dims, sims), chunks=int(sims/dask_chunks))
        elif generator == 'mc-numpy':
            self.generator = lambda: np.random.normal(mean, np.sqrt(delta_t),
                             size=(dims, sims))
        elif generator == 'sobol-np':
            s_gen = sobol_normal(dims, sims)
            self.generator = lambda: (mean+np.sqrt(delta_t)*next(s_gen)).T

        elif generator == 'mc-dask-fast':
            import sys
            sys.path.append('/home/lucio/mypyprojects/ng-numpy-randomstate/')
            from randomstate1.dask.random import normal as xsh128_normal
            self.generator = lambda :xsh128_normal(mean, np.sqrt(delta_t), 
                             size=(dims, sims), chunks=int(sims/dask_chunks))            

        elif generator == 'mc-multithreaded':
            rgen = MultithreadedRNG(dims*sims, seed=seed, threads=n_threads)
            std = np.sqrt(delta_t)
            self.generator =  lambda sims: (mean+std*rgen.fill().values).reshape((dims, sims))
            
      
    def _check_valid_params(self):
        pass


    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = self.generator()
        return out


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

    mean : float
        Mean of the each marginal distribution.

    delta_t : float or fraction
        Time step of the increment from which the variance is derived

    generator: string {'mc-numpy', ''mc-dask'}
        Type of generator.
        mc-numpy: Monte Carlo numpy generator
        mc-dask: Monte Carlo generator that uses dask for parallel calculations

    n_threads: int
        Number of threads to use for multithreaded generator

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
                 'fixed_arrival', 'generator', 'first_generator', 'chunks',
                 'second_generator')

    def __init__(self,  dims, sims_outer, sims_inner, mean, delta_t,
                 mcmv_time, fixed_inner_arrival=True, generator='mc-numpy',
                 n_threads=1, max_prefetch=1, dask_chunks=1, seed=None):
        Rng.__init__(self, dims, sims_outer*sims_inner)
        self.mean = mean
        self.sims_inner = sims_inner
        self.sims_outer = sims_outer
        self.mcmv_time = mcmv_time
        self.fixed_arrival = fixed_inner_arrival
        np.random.seed(seed)

        if generator == 'mc-numpy':
            np.random.seed(seed)
            self.first_generator = lambda sims: np.random.normal(mean, 
                                                    np.sqrt(delta_t),
                                                    size=(dims, sims))
            self.second_generator = self.first_generator
        elif generator == 'mc-dask':
            import dask.array as da
            np.random.seed(seed)
            self.first_generator = lambda sims:np.random.normal(mean, np.sqrt(delta_t),
                         size=(dims, sims))
            chunks = int((sims_outer * sims_inner)/dask_chunks)
            self.chunks = chunks #chunks
            self.second_generator = lambda sims:da.random.normal(mean, np.sqrt(delta_t),
                             size=(dims, sims), chunks=chunks)
        elif generator == 'mc-multithreaded':
            rgen1 = MultithreadedRNG(dims*sims_outer, seed=seed, threads=n_threads)
            std = np.sqrt(delta_t)
            self.first_generator =  lambda sims: (mean+std*rgen1.fill().values).reshape((dims, sims))
            
            
            rgen2 = MultithreadedRNG(dims*sims_outer * sims_inner, 
                                     state=rgen1.rs.get_state(),
                                     threads=n_threads)
            self.second_generator = lambda sims: (mean+std*rgen2.fill().values).reshape((dims, sims))

        elif generator == 'mc-multithreaded-background':
            rgen = BackgroundRNGenerator(dims*sims_outer * sims_inner, seed=seed,
                                         threads=n_threads, max_prefetch=max_prefetch)
            std = np.sqrt(delta_t)
            self.multithr_generator = lambda sims: (mean+std*rgen.generate()).reshape((dims, sims))


        elif generator == 'mc-dask-fast':
            import dask.array as da
            self.first_generator = lambda sims:np.random.normal(mean, np.sqrt(delta_t),
                         size=(dims, sims))

            chunks = int((sims_outer * sims_inner)/dask_chunks)
            self.chunks = chunks #chunks
            import os
            import sys
            parent = os.path.dirname
#            print(os.path.join(parent(parent(parent(parent(__file__)))), 'ng-numpy-randomstate'))
            sys.path.append(os.path.join(parent(parent(parent(parent(__file__)))), 'ng-numpy-randomstate'))

            from randomstate1.dask.random import normal as xsh128_normal
            randomstate1.prng.xoroshiro128plus.RandomState(seed)
            self.second_generator = lambda sims:xsh128_normal(mean, np.sqrt(delta_t),
                             size=(dims, sims), chunks=chunks)
        else:
            raise ValueError('Unknown generator')
        self.generator = self.first_generator

    def _check_valid_params(self):
        pass

    def run_step(self, t):
        if t == 0:
            self.initialize()
        elif t <= self.mcmv_time:
            rn = self.generate(self.sims_outer)
            self.value_t = np.tile(rn, [1,self.sims_inner])
        else:
            self.generator = self.second_generator
            self.value_t = self.generate(self.sims_outer*self.sims_inner)

    def generate(self, sims):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = self.generator(sims)
#        print(np.cov(out))
        return out

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

    def __init__(self, dims, sims_outer, sims_inner, mean, cov,
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
        pass

    def run_step(self, t):
        #self.value_t[...] = self.generate()
        if t <= self.mcmv_time:
            rn = self.generate(self.sims_outer)
            self.value_t = np.tile(rn, [1,self.sims_inner])
        else:
            self.value_t = self.generate(self.sims_outer*self.sims_inner)

    def generate(self, sims):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        out = np.random.multivariate_normal(self.mean, self.cov,
                                            size=sims,
                                            check_valid='raise').T
#        print(out.squeeze()[:2])
#        print(np.cov(out))
        return out


class DimMixGenerator(Rng):
    """class for independent increments of Wiener process

     Parameters
    ----------
    generators: Iterable
        Iterable containing each individual generator
        
    mix_method: string
        Either 'cartesian', meaning that each element generated by one 
        generator will be combined with every possible combination of 
        elements from the other generators or 'direct' where the vectors
        will be simply concatenated
    """
    __slots__ = ('generators')
                 
    def __init__(self,  generators, mix_method):
        self.generators = generators
        self.mix_method = mix_method
    
    def generate(self):
        """Return the next iteration of the random number generator
        """
        res = []
        for gen in self.generators():
            res.append(gen.generate())
        if self.mix_method == 'cartesian':
            ind = [np.arange(len(a)) for a in res]
            comb_ind = np.array(np.meshgrid(*ind))
            rows = [a[i] for a, i in zip(res, comb_ind)]
            out = np.concatenate(rows, axis=-1)
        elif self.mix_method == 'direct':
            out = np.concatenate(res, axis=0)
            
        return out

    
    
class WienerProcess(StochasticVariable):
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

    def run_step_ne(self, t):
        self._evaluate_ne('self_1+dW', out_var='value_t')


class CorrelatedRV(StochasticVariable):
    """class for correlating independent variables

     Parameters
    ----------

     cov: covariance matrix
     rng: random number generator for independent variables with variance = 1
    """
    __slots__ = ('rng', 'm')

    def __init__(self, rng, input_cov, target_cov):
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

    def run_step_ne(self, t):
        """
        This function does not actually use numexpr because the dot product
        is provided by BLAS which is already optimized. However, here we use
        the out parameter which follows the logic of other run_step_ne, while
        run_step normally creates copies. Also run_step_ne can rely on
        value_t having been initialized to the proper size
        """
        np.dot(self.m, self.rng(), out=self.value_t)

class PreCalculatedFeed(StochasticVariable):
    """class for feeding a precalculated stream (usually quadrature points
    for dW)

    points: 3-d array. dimensions are: time, variable, sample
    """

    __slots__ = ('points', 'col', 'sims')

    def __init__(self, points):
        self.points = points
        self.sims = points.shape[2]
        self.col = 0

    def run_step(self, t):
        self.value_t = self.points[self.col, :,:]
        self.col += 1
        