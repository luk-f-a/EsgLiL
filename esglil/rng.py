#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:27:12 2017

@author: luk-f-a
"""

import xarray as xr
import numpy as np

class Rng(object):
    """Base class for random number generators
    """
    __slots__ = ('shape', '_use_xr', '_to_xr', '_loop_dim',
                 '_dims', '_coords', 'type')

    def __init__(self, shape, svar_names=None, loop_dim=None):
        assert loop_dim is None or type(loop_dim) is str
        if type(loop_dim) is str:
            assert loop_dim in ('svar', 'sim', 'time')
        if isinstance(shape, tuple):
            self.shape = shape
            self._use_xr = False
        elif isinstance(shape, dict):
            self._use_xr = True
            self.shape = [shape['svar'], shape['sim'], shape['time']]
            self._dims = ['svar','sim', 'time']
            if svar_names is None:
                svars = ['rv_{}'.format(i) for i in range(1, shape['svar']+1)]
            else:
                svars = svar_names
            self._coords = {'svar': svars,
                            'sim': range(1, shape['sim']+1),
                            'time': range(1, shape['time']+1)}
            #self._to_xr = lambda data, me: xr.DataArray(data,
            #                          dims=me._dims,
            #                          coords=me._coords)
            if loop_dim:
                self._loop_dim = loop_dim
                self._coords[loop_dim] = list(self._coords[loop_dim])
            else:
                self._loop_dim = None
            
 
                
    def __iter__(self):
        return self

    def __next__(self):
            return self.generate()

    def _fwd_coords(self):
        self._coords[self._loop_dim] = [x+len(self._coords[self._loop_dim])
                                        for x in
                                        self._coords[self._loop_dim]]
        
    def generate(self):
        """Implement here the code to provide the next iteration of the
        random number generator
        """
        raise NotImplementedError
        
    @property
    def svars(self):
        if self._use_xr:
            return self._coords['svar']
        else:
            return [None]


class UniformRng(Rng):
    """class for uniform random number generation

     Parameters
    ----------
    shape : tuple or dict
        If a tuble, the class will return numpy arrays of this shape
        If a dict, it should contain the desired name of dimensions as keys
        and the size of the dimensions (per request to __next__ or generate)
        as values.
        
    svar_names: list, optional
        Names of each output stochastic variable. If not given, will be 
        called "rv_[number]", with number ranging from 1 to max(svar)   

    loop_dim : string or None
        Ignored is shape is a tuple. If a loop_dim is provided,
        the instance will keep track of that dimension and increase the
        coordinates in each succesive call.
    """
    def generate(self):
        """Return the next iteration of the random number generator
       

        """
        np_rnd = np.random.rand(*self.shape)
        if self._use_xr:
            #out = self._to_xr(np_rnd, self)
            out = xr.DataArray(np_rnd, dims=self._dims, coords=self._coords)
            if self._loop_dim is not None:
                self._fwd_coords()
        else:
            out = np_rnd
        return out


class NormalRng(Rng):
    """class for normal random number generation

     Parameters
    ----------
    shape : tuple or dict
        If a tuble, the class will return numpy arrays of this shape
        If a dict, it should contain the desired name of dimensions as keys
        and the size of the dimensions (per request to __next__ or generate)
        as values.
        
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
        
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. 
        It must be symmetric and positive-semidefinite for proper sampling.
    
    svar_names: list, optional
        Names of each output stochastic variable. If not given, will be 
        called "rv_[number]", with number ranging from 1 to max(svar)

    loop_dim : string or None
        Ignored is shape is a tuple. If a loop_dim is provided,
        the instance will keep track of that dimension and increase the
        coordinates in each succesive call.
        
    """
    __slots__ = ('mean', 'cov')
                 
    def __init__(self, shape, mean, cov, svar_names=None, loop_dim=None):
        Rng.__init__(self, shape=shape, svar_names=svar_names,
                     loop_dim=loop_dim)
        self.mean = mean
        self.cov = cov
        
    def _check_valid_params(self):
        #TODO: if output is numpy, mean must be size 1 and cov 1x1
        # if output is xr size of mean and cov must agree with svar dim
        pass
    
    def _reshape_array(self, a):
        #TODO: the shape of array a needs to be aligned to the dims
        return a
        
    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        
        if self._use_xr:
            np_rnd = np.random.multivariate_normal(self.mean, self.cov, 
                                               self.shape[1:],
                                               check_valid='raise')
            np_rnd = np.transpose(np_rnd, axes=(2,0,1))
            out = xr.DataArray(np_rnd, dims=self._dims, coords=self._coords)
            if self._loop_dim is not None:
                self._fwd_coords()
        else:
            np_rnd = np.random.multivariate_normal(self.mean, self.cov, 
                                               self.shape,
                                               check_valid='raise')
            out = np_rnd.squeeze()
        return out