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
import queue as Queue
import threading

class MultithreadedRNG(object):
    def __init__(self, n, seed=None, threads=None):
        #rs = randomstate.prng.xorshift1024.RandomState(seed)
        import randomstate
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

    def __next__(self):
        self.fill()
        return self.values

    def __iter__(self):
        return self
    
    def __del__(self):
        self.executor.shutdown(False)
        


class BackgroundRNGenerator(threading.Thread):
    def __init__(self, n, seed=None, threads=1, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = MultithreadedRNG(n, seed=seed, threads=threads)
        self.daemon = True
        self.start()


    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def generate(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

### The following process based generator was very slow due to the inter-process communication
        
from multiprocessing import Process, Manager
from multiprocessing import Queue as mQueue
class BackgroundProcessRNGenerator():
    def __init__(self, n, threads, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        self.process = Process(target=self.run)
        self.queue = Manager().Queue(max_prefetch)
        self.generator = MultithreadedRNG(n, threads=threads)
        
        self.process.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def generate(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item
    
#class MultithreadedRNG_old(object):
#    def __init__(self, size, mean=0, stdev=1, seed=None, threads=None):
#        rs = randomstate.prng.xorshift1024.RandomState(seed)
#        if threads is None:
#            threads = multiprocessing.cpu_count()
#        self.threads = threads
#
#        self._random_states = []
#        for _ in range(0, threads-1):
#            _rs = randomstate.prng.xorshift1024.RandomState()
#            _rs.set_state(rs.get_state())
#            self._random_states.append(_rs)
#            rs.jump()
#        self._random_states.append(rs)
#
#        self.mean = mean
#        self.stdev = stdev
#        if isinstance(size, int):
##            self.n = size
#            self.size = (size,)
#            self.values = np.empty(size)
#            self.step = np.ceil(size / threads).astype(np.int)
#        elif isinstance(size, tuple):
#            assert len(size) == 2 or len(size) == 1
#            if len(size) == 2:
#                n = size[1]
#            else:
#                n = size[0]
#            self.size = size
#            self.values = np.empty(size)
#            self.step = np.ceil(n / threads).astype(np.int)            
#
#        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
#
#    def fill(self):
#        def _fill(random_state, out, first, last):
#            random_state.standard_normal(out=out[...,first:last])
#            out[...,first:last] *= self.stdev
#            out[...,first:last] += self.mean
#            
#        futures = {}
#        for i in range(self.threads):
#            args = (_fill, self._random_states[i], self.values, i * self.step, (i + 1) * self.step)
#            futures[self.executor.submit(*args)] = i
#        concurrent.futures.wait(futures)
#        return self
#
#    def __del__(self):
#        self.executor.shutdown(False)
