#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:21:54 2017

@author: luk-f-a
"""

from fractions import Fraction
import numpy as np

class ESG(object):
    __slots = ['eq', 'clock', 'dt_sim']
    def __init__(self, dt_sim, **models):
        self.eq = models
        self.clock = 0
        self.dt_sim = Fraction(1,int(round(1/dt_sim,0)))
        
        
    def run_step(self, t=None):
        if t is None:
            steps = 1
        else:
            steps = int(round((t-self.clock)/self.dt_sim,0))
        for _ in range(steps):
            self.clock += self.dt_sim
            for model in self.eq:
                self.eq[model].run_step(float(self.clock))
    
    def full_run(self, dt_out, max_t):
        dt_out = Fraction(dt_out)
        assert float(dt_out/self.dt_sim) % 1 == 0
        out = {}
        for ts in range(1, int(float(max_t/self.dt_sim))+1):
            self.run_step()
            t = float(ts*self.dt_sim)
            if ts % float(dt_out/self.dt_sim) == 0:
                out[t] = self.value_t
        return out
        
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
                digits = len(str(val.shape[0]))
                for d in range(val.shape[0]):
                    value_dict[model+'_'+str(d).zfill(digits)] = val[d,:].squeeze()
        return pd.DataFrame(value_dict)
    
    @property
    def value_t(self):
        out = {}
        for model in self.eq: 
            model_out = self.eq[model].value_t
            if type(model_out) is dict:
                out.update(model_out)
            else:
                out[model] = model_out
        return out
    
    def __getitem__(self, key):
        return self.eq[key]
#    def __call__(self):
#        return {model:self.eq[model]() for model in self.eq}