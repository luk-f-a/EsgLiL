#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:15:52 2018

@author: luk-f-a

Taken from 
http://bashtage.github.io/ng-numpy-randomstate/doc/multithreading.html
"""

import randomstate
import multiprocessing
import concurrent.futures
import numpy as np


class MultithreadedRNG(object):
    def __init__(self, n, seed=None, threads=None):
        #rs = randomstate.prng.xorshift1024.RandomState(seed)
        rs = randomstate.prng.xoroshiro128plus.RandomState(seed)
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        self._random_states = []
        for _ in range(0, threads-1):
            _rs = randomstate.prng.xoroshiro128plus.RandomState()
            _rs.set_state(rs.get_state())
            self._random_states.append(_rs)
            rs.jump()
        self._random_states.append(rs)

        self.n = n
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(n)
        self.step = np.ceil(n / threads).astype(np.int)

    def fill(self):
        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last], method='zig')
        futures = {}
        for i in range(self.threads):
            args = (_fill, self._random_states[i], self.values, i * self.step, (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        return self

    def __del__(self):
        self.executor.shutdown(False)
        

class MultithreadedRNG_old(object):
    def __init__(self, size, mean=0, stdev=1, seed=None, threads=None):
        rs = randomstate.prng.xorshift1024.RandomState(seed)
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        self._random_states = []
        for _ in range(0, threads-1):
            _rs = randomstate.prng.xorshift1024.RandomState()
            _rs.set_state(rs.get_state())
            self._random_states.append(_rs)
            rs.jump()
        self._random_states.append(rs)

        self.mean = mean
        self.stdev = stdev
        if isinstance(size, int):
#            self.n = size
            self.size = (size,)
            self.values = np.empty(size)
            self.step = np.ceil(size / threads).astype(np.int)
        elif isinstance(size, tuple):
            assert len(size) == 2 or len(size) == 1
            if len(size) == 2:
                n = size[1]
            else:
                n = size[0]
            self.size = size
            self.values = np.empty(size)
            self.step = np.ceil(n / threads).astype(np.int)            

        self.executor = concurrent.futures.ThreadPoolExecutor(threads)

    def fill(self):
        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[...,first:last])
            out[...,first:last] *= self.stdev
            out[...,first:last] += self.mean
            
        futures = {}
        for i in range(self.threads):
            args = (_fill, self._random_states[i], self.values, i * self.step, (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        return self

    def __del__(self):
        self.executor.shutdown(False)
