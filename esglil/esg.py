#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:21:54 2017

@author: luk-f-a
"""

from fractions import Fraction
import numpy as np

class ESG(object):
    def __init__(self, dt_sim, **models):
        self.eq = models
        self.clock = 0
        self.dt_sim = Fraction(dt_sim)
        
        
    def run_step(self):
        self.clock += self.dt_sim
        for model in self.eq:
            self.eq[model].run_step(float(self.clock))
            
    def full_run_to_pandas(self, dt_out, max_t):
        import pandas as pd
        dt_out = Fraction(dt_out)
        assert float(dt_out/self.dt_sim) % 1 == 0
        out = {}
        for ts in range(1, int(float(max_t/self.dt_sim))+1):
            self.run_step()
            t = float(ts*self.dt_sim)
            if ts % float(dt_out/self.dt_sim) == 0:
                out[t] = self.df_value_t()
                out[t].index.names = ['sim']
        df = pd.concat(out, axis=0)
        df.index.names = ['time', 'sim']
        return df
    
    def df_value_t(self):
        import pandas as pd
        value_dict = {}
        for model in self.eq:
            val = np.array(self.eq[model].value_t).squeeze()
            if len(val.shape)<=1:
                value_dict[model] = val
            else:
                for d in range(val.shape[1]):
                    value_dict[model+'_'+str(d).zfill(2)] = val[...,d].squeeze()
        return pd.DataFrame(value_dict)
    
    @property
    def value_t(self):
        return {model: self.eq[model].value_t for model in self.eq}
#    def __call__(self):
#        return {model:self.eq[model]() for model in self.eq}