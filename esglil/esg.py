#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:21:54 2017

@author: luk-f-a
"""

from fractions import Fraction
import numpy as np


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:21:54 2017

@author: luk-f-a
"""

from fractions import Fraction
import numpy as np
import pandas as pd
from itertools import repeat
from esglil.common import SDE

class Model(object):
    """ Performs time loop on objects associated with it
    clock measures the last time point equations (not Models) directly associated with this Model were run
    dense clock measure the last time point other Models inside this model were run
    The difference between them happens when the outside Model has a delta bigger than
    the inner Model.
    While time is passing inside the inside Model
    """
    __slots = ['eq', 'clock', 'dense_clock', 'dt_sim', 'is_model_loop']    
    def __init__(self, dt_sim, **models):
        self.eq = models
        self.clock = 0
        self.dense_clock = 0
        self.dt_sim = float_to_fraction(dt_sim)
        for eq in models:
            if hasattr(eq, 'dt_sim'):
                assert dq.dt_sim < dt_sim, "Cannot add modelloop with larger timestep"
        
        
    def run_step(self, t=None):
        if t is None:
            steps = 1
        else:
            steps = int(round((t-self.clock)/self.dt_sim,0))
        for _ in range(steps):
            self.clock += self.dt_sim
            for model in self.eq:
                if isinstance(self.eq[model], bool):
                    print(model)
                self.eq[model].run_step(float(self.clock))
   
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

    def _replace_links(self, name, old_object, new_object):
        for m in self.eq:
            model = self.eq[m]
            if isinstance(model, Model):
                model[name] = new_object
            elif isinstance(model, SDE):
                model._replace_links(name, old_object, new_object)
                
    def __getitem__(self, key):
        try:
            out = self.eq[key]
        except:
            out = None
            for m in self.eq:
                model = self.eq[m]
                if isinstance(model, Model):
                    try:
                        out = model[key]
                    except:
                        pass
            if out is None:
                raise KeyError
        return out

    def __setitem__(self, key, item):
        """ replace one element in the computational tree and including 
        rebuilding the edges to all dependent objects
        """
        old_object = self.__getitem__(key)
        for m_name in self.eq:
            if m_name == key:
                self.eq[m_name] = item
            else:
                model = self.eq[m_name]
                if isinstance(model, Model):
                    model[key] = item
                elif isinstance(model, SDE):
                    model._replace_links(key, old_object, item)                
                
            
    
    def full_run(self, dt_out, max_t):
        dt_out = Fraction(dt_out)
        assert float(dt_out/self.dt_sim) % 1 == 0
        out = {}
        for ts in range(1, int(float(max_t/self.dt_sim))+1):
            self.run_step()
            t = float(ts*self.dt_sim)
            if ts % float(dt_out/self.dt_sim) == 0:
                out[self.clock] = self.value_t
        return out
    
        
#    def run_multistep_to_pandas_old(self, dt_out, max_t, out_vars=None):
#        import pandas as pd
#        dt_out = float_to_fraction(dt_out)
#        assert max_t > float(self.clock)
#        if dt_out < self.dt_sim:
#            assert self.dt_sim % dt_out == 0
#            return self._dense_run_to_pandas(dt_out, max_t, out_vars)
#        else:
#            assert dt_out % self.dt_sim == 0
#            return self._sparse_run_to_pandas(dt_out, max_t, out_vars)

    def run_multistep_to_pandas(self, dt_out, max_t, out_vars=None):

        if isinstance(out_vars, list):
            out_vars = {var:slice(None) for var in out_vars}
        dt_out = float_to_fraction(dt_out)
        assert max_t > float(self.clock)
        out = []
        nb_steps = int(round(float((max_t-self.clock)/self.dt_sim),0))
        for ts in range(1, nb_steps+1):
            self.clock += self.dt_sim
            for model in self.eq:
                if isinstance(self.eq[model], Model):
                    df_out = self.eq[model].run_multistep_to_pandas(dt_out, 
                           self.clock, out_vars=out_vars)
                    out.append(df_out)
                else:
                    self.eq[model].run_step(float(self.clock))
#            if dt_out <= self.dt_sim or ts % float(dt_out/self.dt_sim) == 0:        
            if dt_out <= self.dt_sim or self.clock % float(dt_out) == 0:        
                out.append(self.df_value_t(out_vars))
        if len(out)==0:
            df = None
        else:
            out = [o for o in out if o is not None]
            df = pd.concat(out, axis=1)
        return df


    def run_multistep_to_dict(self, dt_out, max_t, out_vars=None):
        if isinstance(out_vars, list):
            out_vars = {var:slice(None) for var in out_vars}
        dt_out = float_to_fraction(dt_out)
        assert max_t > float(self.clock)
        out = {}
        nb_steps = int(round(float((max_t-self.clock)/self.dt_sim),0))
        for ts in range(1, nb_steps+1):
            self.clock += self.dt_sim
            for model in self.eq:
                if isinstance(self.eq[model], Model):
                    dict_out = self.eq[model].run_multistep_to_dict(dt_out, 
                           self.clock, out_vars=out_vars)
                    out.update(dict_out)
                else:
                    self.eq[model].run_step(float(self.clock))
#            if dt_out <= self.dt_sim or ts % float(dt_out/self.dt_sim) == 0:        
            if dt_out <= self.dt_sim or self.clock % float(dt_out) == 0:
                d_val_t = self.dict_value_t(out_vars)
                if self.clock in out:
                    out[self.clock].update(d_val_t[self.clock])
                else:
                    out.update(d_val_t)
        return out
        
#    def _sparse_run_to_pandas(self, dt_out, max_t, out_vars=None):
#        """This method makes a run blah blah
#        """
#        out = []
#        nb_steps = int(round(float((max_t-self.clock)/self.dt_sim),0))
#        for ts in range(1, nb_steps+1):
#            self.run_step()
#            if ts % float(dt_out/self.dt_sim) == 0:
#                out.append(self.df_value_t(out_vars))
#
#        df = pd.concat(out, axis=1)
#        return df
#    
#    def _dense_run_to_pandas(self, dt_out, max_t, out_vars=None):
#        """
#        This method allows for a dt_out which is smaller than the dt of the 
#        model, thus allowing to record the full history of sub-models running on
#        a more granular timegrid
#        """
#        out = []
#        nb_steps = int(round(max_t/dt_out,0))
#
#        for ts in range(1, nb_steps+1):
#            t = ts*dt_out
#
#            if t % self.dt_sim == 0:
#                self.run_step()
#                out.append(self.df_value_t(out_vars))
#            else:
#                for model in self.eq:
#                    if hasattr(self.eq[model], 'run_multistep_to_pandas'):
#                        mstep_fc = self.eq[model].run_multistep_to_pandas
#                        m_df = mstep_fc(dt_out, max_t=t, out_vars=out_vars)
##                        m_df = m_df.rename(index={orig:model+'_'+orig 
##                                             for orig in m_df.index.get_level_values('model')})
#                        out.append(m_df)
#                        
#        df = pd.concat(out, axis=1)
#        return df

    def df_value_t(self, out_vars):
        import pandas as pd
        def get_this_model_step(name):
            if (out_vars is None):
                return True
            if not name in out_vars:
                return False
            else:
                if (((out_vars[name].start is None) or 
                    out_vars[name].start <= self.clock) and
                    ((out_vars[name].stop is None) or 
                out_vars[name].stop >= self.clock)):
                        return True
                else:
                    return False
                
        value_dict = {}
        df_list = []
        for model in self.eq:
            if isinstance(self.eq[model], Model):
                continue
            else:
                model_val = self.eq[model].value_t
                if isinstance(model_val, dict):
                    for sub_output in model_val:
                        if get_this_model_step(model) or get_this_model_step(sub_output):
                            val = model_val[sub_output]
                            if get_this_model_step(model):
                                key_name = model + "_" + str(sub_output)
                            else:
                                key_name = sub_output
                            v_d = value_to_dictionary(key_name, val)
                            value_dict.update(v_d)
                else:
                    if get_this_model_step(model):
                        value_dict.update(value_to_dictionary(model, model_val))
        try:
            df = pd.DataFrame.from_dict(value_dict)
        except:
            print(value_dict)
            raise
        tuples = list(zip(df.columns, repeat(self.clock)))
        index = pd.MultiIndex.from_tuples(tuples, names=['model', 'time'])
        df.columns = index
        df.index.names = ['sim']
        return df
        
    def dict_value_t(self, out_vars):

        def get_this_model_step(name):
            if (out_vars is None):
                return True
            if not name in out_vars:
                return False
            else:
                if (((out_vars[name].start is None) or 
                    out_vars[name].start <= self.clock) and
                    ((out_vars[name].stop is None) or 
                out_vars[name].stop >= self.clock)):
                        return True
                else:
                    return False

        dict_out = {}
        for model in self.eq:
            if isinstance(self.eq[model], Model):
                continue
            else:
                model_val = self.eq[model].value_t
                if isinstance(model_val, dict):
                    for sub_output in model_val:
                        if get_this_model_step(model) or get_this_model_step(sub_output):
                            val = model_val[sub_output]
                            if get_this_model_step(model):
                                key_name = model + "_" + str(sub_output)
                            else:
                                key_name = sub_output
                            dict_out[key_name] = val
                else:
                    if get_this_model_step(model):
                        dict_out[model] = model_val
        out = {self.clock: dict_out}
        return out

        
    
def value_to_dictionary(model_name, val):
    value_dict = {}
    val = np.array(val).squeeze()
    if len(val.shape)<=1:
        value_dict[model_name] = val
    else:
        digits = len(str(val.shape[0]))
        for d in range(val.shape[0]):
            name = model_name+'_'+str(d).zfill(digits)
            value_dict[name] = val[d,:].squeeze()
    return value_dict

def float_to_fraction(flt):
    return Fraction(1,int(round(1/flt,0)))

ESG = Model