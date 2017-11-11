#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:16:16 2017

@author: luk-f-a
"""

from . import rng
from . import pipeline
from . import equity_models
import numpy as np
import xarray as xr
from fractions import Fraction

class  GBM3(object):

    
    def __init__(self, sims=1, max_t=10, loop_time=True, mu=0, sigma=1, 
                 delta_t=1, time_sampling_ratio=1):
        
        self.max_t =  max_t
        assert type(loop_time) is bool
        if loop_time:
            loop_dim = 'time'
        else:
            loop_dim = None
        dW = rng.NormalRng(shape={'svar':1, 'sim':sims, 
                                  'time':max_t*time_sampling_ratio},
                           mean=[0], cov=[[delta_t/time_sampling_ratio]],
                                 loop_dim=loop_dim)
        gbm = equity_models.GeometricBrownianMotion(mu=mu, sigma=sigma, 
                            s_zero=100, delta_t_out=delta_t, 
                            delta_t_in=Fraction(delta_t, time_sampling_ratio))
    
        self.ppl = pipeline.Pipeline([dW, gbm])
        
    def time_loop_gen(self):
        return (self.ppl.generate().squeeze() for i in range(self.max_t))
       
           
    def generate(self):
        return self.ppl.generate().squeeze()
    
class  GBM2(object):
    def __new__(cls, *args, **kwargs):
        if 'loop_time' in kwargs:
            if kwargs['loop_time']:
                instance = super(GBM2, cls).__new__(cls)
                instance.__init__(*args, **kwargs)
                return instance
                
            else:
                instance = super(GBM2, cls).__new__(cls)
                instance.__init__(*args, **kwargs)
                return instance()
    
    def __init__(self, sims=1, max_t=10, loop_time=True, mu=0, sigma=1, delta_t=1):
        
        (self.sims, self.max_t, self.loop_time, 
         self.mu, self.sigma, self.delta_t) = (sims, max_t, loop_time, 
                                               mu, sigma, delta_t)
        if loop_time:
            loop_dim = 'time'
        else:
            loop_dim = None
        dW = rng.NormalRng(shape={'svar':1, 'sim':sims, 'time':max_t},
                           mean=[0], cov=[[delta_t]], loop_dim=loop_dim)
        gbm = equity_models.GeometricBrownianMotion(mu=mu, sigma=sigma, 
                                                    s_zero=100, delta_t=delta_t)
    
        self.ppl = pipeline.Pipeline([dW, gbm])
        
    def __iter__(self):
        for i in range(self.max_t):
            x = self.ppl.generate().squeeze()
            yield x
            
    def __call__(self):
        gen = iter(self)
        return next(gen)
    
    
def GBM(sims=1, max_t=10, loop_time=True, mu=0, sigma=1, delta_t=1):
    if loop_time:
        loop_dim = 'time'
    else:
        loop_dim = None
    dW = rng.NormalRng(shape={'svar':1, 'sim':sims, 'time':max_t},
                       mean=[0], cov=[[delta_t]], loop_dim=loop_dim)
    gbm = equity_models.GeometricBrownianMotion(mu=mu, sigma=sigma, 
                                                s_zero=100, delta_t=delta_t)

    ppl = pipeline.Pipeline([dW, gbm])
    

    if loop_time:
        print('generator')
        yield ppl.generate()
    else:
        print('function')
        return ppl.generate().squeeze()
    