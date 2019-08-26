#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:15:52 2018

@author: luk-f-a

Taken from 
http://bashtage.github.io/ng-numpy-randomstate/doc/multithreading.html
"""


import multiprocessing
import concurrent.futures
import numpy as np
from numpy.random import Generator, PCG64

class MultithreadedRNG(object):
    def __init__(self, n, seed=None, threads=None):
        rg = PCG64(seed)
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        self._random_generators = [rg]
        last_rg = rg
        for _ in range(0, threads-1):
            new_rg = last_rg.jumped()
            self._random_generators.append(new_rg)
            last_rg = new_rg

        self.n = n
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(n)
        self.step = np.ceil(n / threads).astype(np.int)

    def fill(self):
        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last])

        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    self.values,
                    i * self.step,
                    (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)

    def __del__(self):
        self.executor.shutdown(False)