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
                 '_dims', '_coords')

    def __init__(self, shape, loop_dim=None):
        if isinstance(shape, tuple):
            self.shape = shape
            self._use_xr = False
        elif isinstance(shape, dict):
            self._use_xr = True
            self.shape = tuple(shape.values())
            self._dims = ['sim', 'svar', 'time']
            self._coords = {'sim': range(1, shape['sim']+1),
                            'svar': range(1, shape['svar']+1),
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


class UniformRng(Rng):
    """class for uniform random number generation

     Parameters
    ----------
    shape : tuple or dict
        If a tuble, the class will return numpy arrays of this shape
        If a dict, it should contain the desired name of dimensions as keys
        and the size of the dimensions (per request to __next__ or generate)
        as values.

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
    """class for uniform random number generation

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

    loop_dim : string or None
        Ignored is shape is a tuple. If a loop_dim is provided,
        the instance will keep track of that dimension and increase the
        coordinates in each succesive call.
        
    """
    __slots__ = ('mean', 'cov')
                 
    def __init__(self, shape, mean, cov, loop_dim=None):
        Rng.__init__(self, shape, loop_dim)
        self.mean = mean
        self.cov = cov
        
    def generate(self):
        """Return the next iteration of the random number generator
        """
        np_rnd = np.random.multivariate_normal(self.mean, self.cov, self.shape,
                                               check_valid='raise')
        if self._use_xr:
            #out = self._to_xr(np_rnd, self)
            out = xr.DataArray(np_rnd, dims=self._dims, coords=self._coords)
            if self._loop_dim is not None:
                self._fwd_coords()
        else:
            out = np_rnd
        return out