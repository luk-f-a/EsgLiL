#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:21:54 2017

@author: luk-f-a
"""

from fractions import Fraction

class ESG(object):
    def __init__(self, dt_sim, dt_out, **models):
        self.eq = models
        self.clock = 0
        self.dt_sim = Fraction(dt_sim)
        self.dt_out = Fraction(dt_out)
        assert float(self.dt_out/self.dt_sim) % 1 == 0
        
    def run_step(self):
        self.clock += self.dt_sim
        for model in self.eq:
            self.eq[model].run_step(float(self.clock))
            
    def full_run_to_pandas(self, max_t):
        import pandas as pd
        out = {}
        for ts in range(1, int(float(max_t/self.dt_sim))+1):
            self.run_step()
            t = float(ts*self.dt_sim)
            if ts % float(self.dt_out/self.dt_sim) == 0:
                out[t] = pd.DataFrame(self.value_t)
                out[t].index.names = ['sim']
        return pd.concat(out, axis=0)
    
    @property
    def value_t(self):
        return {model: self.eq[model].value_t for model in self.eq}
#    def __call__(self):
#        return {model:self.eq[model]() for model in self.eq}