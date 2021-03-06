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
from esglil.common import FunctionOfVariable
from collections import Iterable
from typing import Dict

class Model(object):
    """ Performs time loop on objects associated with it.
    
     Parameters
    ----------
    dt_sim: float or Fraction
        time step of the simulation
    
    **models: models to be included in the loop, identified by name using
        their keyword argument.
    """
    __slots = ['eq', 'clock', 'dense_clock', 'dt_sim', 'is_model_loop',
               'initialized']
    def __init__(self, dt_sim, **models):
        self.eq = models
        self.clock = 0
        self.dt_sim = float_to_fraction(dt_sim)
        self.initialized = False
        self.check_dt_sim(models, dt_sim)

    @classmethod
    def check_dt_sim(cls, models, this_dt_sim):
        for eq in models:
#            if hasattr(eq, 'dt_sim'):
            if isinstance(eq, cls):
                assert eq.dt_sim < this_dt_sim, ("Cannot add model loop with "
                                                 "larger timestep")        
    def initialize(self):
        if not self.initialized:
            self.run_step(0)

    def run_step(self, t=None):
        if t is None:
            steps = 1
        elif t == 0:
            self.initialized = True
            steps = 1
        else:
            steps = int(round((t-self.clock)/self.dt_sim, 0))

        for _ in range(steps):
            if t > 0:
                self.clock += self.dt_sim
            for model in self.eq:
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
            elif isinstance(model, FunctionOfVariable):
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
                elif isinstance(model, FunctionOfVariable):
                    model._replace_links(old_object, item)


    def __delitem__(self, key):
        del self.eq[key]

    def run_multistep_to_pandas(self, dt_out, max_t, out_vars=None):
        """Runs the models step-by-step (including all submodels) from
        the current time (as indicated in self.clock) until max_t and
        returns the variables requested in out_vars in a pandas dataframe
        
        Parameters
        ----------     
        dt_out: float or Fraction
            the timestep of the output. This is different from the simulation
            timestep of the model. It can be smaller, useful when there are
            submodels running with a shorter timestep, equal or larger, when
            only longer intervals are needed (e.g. annual output with weekly
            simulations)
            
        max_t: float or Fraction
            time until which to run the run the simulation
            
        out_vars: list or dictionary
            if a list, is a list of variable names that will be recorded
            in each step (dt_out step)
            if a dictionary, the key is the model name to be included at the
            value must be a slice object.
            Examples:
                {'a':slice(5)} means include all output of variable 'a'
                until time 5 (including 5)
                {'a':slice(None)} means include all output of variable 'a',
                for every timestep
                {'a':slice(1, 5)} means include all output of variable 'a'
                from 1 until time 5 (including 1 and 5)    
                {'a':slice(1, 5), 'b':slice(2, 4)}  include output of variables
                'a' and 'b' each according to its associated slice.
                
        """
        import warnings
        warnings.warn("Output to pandas is being considered for deprecation", 
                      DeprecationWarning)
        if isinstance(out_vars, list):
            out_vars = {var:slice(None) for var in out_vars}
        dt_out = float_to_fraction(dt_out)
        assert max_t > float(self.clock)
        out = []
        nb_steps = int(round(float((max_t-self.clock)/self.dt_sim), 0))
        for ts in range(1, nb_steps+1):
            self.clock += self.dt_sim
            for model in self.eq:
                if isinstance(self.eq[model], Model):
                    df_out = self.eq[model].run_multistep_to_pandas(dt_out,
                                                self.clock, out_vars=out_vars)
                    out.append(df_out)
                else:
                    self.eq[model].run_step(float(self.clock))
            if dt_out <= self.dt_sim or self.clock % float(dt_out) == 0:
                out.append(self.df_value_t(out_vars))
        if len(out) == 0:
            df = None
        else:
            out = [o for o in out if o is not None]
            df = pd.concat(out, axis=1)
        return df

    def run_multistep_to_dict(self, dt_out, max_t, out_vars=None,
                              use_numexpr=False):
        """Runs the models step-by-step (including all submodels) from
        the current time (as indicated in self.clock) until max_t and
        returns the variables requested in out_vars in a nested dictionary with 
        structure {time1: {model1: values, model2: values},
                   time2: {model1: values, model2: values}}
        
        Parameters
        ----------     
        dt_out: float or Fraction
            the timestep of the output. This is different from the simulation
            timestep of the model. It can be smaller, useful when there are
            submodels running with a shorter timestep, equal or larger, when
            only longer intervals are needed (e.g. annual output with weekly
            simulations)
            
        max_t: float or Fraction
            time until which to run the run the simulation
            
        out_vars: list or dictionary
            if a list, is a list of variable names that will be recorded
            in each step (dt_out step)
            if a dictionary, the key is the model name to be included at the
            value must be a slice object.
            Examples:
                {'a':slice(5)} means include all output of variable 'a'
                until time 5 (including 5)
                'a':slice(5,5)} means include output of variable 'a'
                only for time 5
                {'a':slice(None)} means include all output of variable 'a',
                for every timestep
                {'a':slice(1, 5)} means include all output of variable 'a'
                from 1 until time 5 (including 1 and 5)    
                {'a':slice(1, 5), 'b':slice(2, 4)}  include output of variables
                'a' and 'b' each according to its associated slice.
         
        use_numexpr: boolean
            If True, the method run_step_ne of each model will be called,
            trigerring (in available) a calculation using NumExpr. If not 
            available, the method run_step will be used.
            This option cannot be used with Dask random number generators
        """
        if isinstance(out_vars, list):
            out_vars = {var:slice(None) for var in out_vars}
        if use_numexpr and not self.initialized:
            self.initialize()
        dt_out = float_to_fraction(dt_out)
        assert max_t > float(self.clock)
        out = {}
        nb_steps = int(round(float((max_t-self.clock)/self.dt_sim), 0))
        for ts in range(1, nb_steps+1):
            self.clock += self.dt_sim
            for model in self.eq:
                if isinstance(self.eq[model], Model):
                    dict_out = self.eq[model].run_multistep_to_dict(dt_out,
                                              self.clock, out_vars=out_vars,
                                              use_numexpr=use_numexpr)
                    out.update(dict_out)
                else:
                    if use_numexpr and hasattr(self.eq[model], 'run_step_ne'):
                        self.eq[model].run_step_ne(float(self.clock))
                    else:
                        self.eq[model].run_step(float(self.clock))
            if dt_out <= self.dt_sim or self.clock % float(dt_out) == 0:
                d_val_t = self.dict_value_t(out_vars)
                if self.clock in out:
                    out[self.clock].update(d_val_t[self.clock])
                else:
                    out.update(d_val_t)
        return out

    def df_value_t(self, out_vars):
        import pandas as pd
        get_this_step = lambda name: self.get_this_model_step(name, out_vars)

        value_dict = {}
        df_list = []
        for model in self.eq:
            if isinstance(self.eq[model], Model):
                continue
            else:
                model_val = self.eq[model].value_t
                if isinstance(model_val, dict):
                    for sub_output in model_val:
                        if (get_this_step(model) or
                                get_this_step(sub_output)):
                            val = model_val[sub_output]
                            if get_this_step(model):
                                key_name = model + "_" + str(sub_output)
                            else:
                                key_name = sub_output
                            v_d = value_to_dictionary(key_name, val)
                            value_dict.update(v_d)
                else:
                    if get_this_step(model):
                        value_dict.update(value_to_dictionary(model, model_val))
        try:
            df = pd.DataFrame.from_dict(value_dict)
        except:
            print('Conversion to pandas failed with:', value_dict)
            raise
        tuples = list(zip(df.columns, repeat(self.clock)))
        index = pd.MultiIndex.from_tuples(tuples, names=['model', 'time'])
        df.columns = index
        df.index.names = ['sim']
        return df


    def get_this_model_step(self, name, out_vars):
        if out_vars is None:
            return True
        if not name in out_vars:
            return False
        else:
            if isinstance(out_vars[name], Iterable):
                return self.clock in out_vars[name]
            else:
                if (((out_vars[name].start is None) or
                    out_vars[name].start <= self.clock) and
                    ((out_vars[name].stop is None) or
                out_vars[name].stop >= self.clock)):
                    return True
                else:
                    return False 
        
    def dict_value_t(self, out_vars):
        get_this_step = lambda name: self.get_this_model_step(name, out_vars)
        dict_out = {}
        for model in self.eq:
            if isinstance(self.eq[model], Model):
                continue
            else:
                model_val = self.eq[model].value_t
                if isinstance(model_val, dict):
                    for sub_output in model_val:
                        if (get_this_step(model) or
                            get_this_step(sub_output)):
                            val = model_val[sub_output]
                            if get_this_step(model):
                                key_name = model + "_" + str(sub_output)
                            else:
                                key_name = sub_output
                            dict_out[key_name] = val
                else:
                    if get_this_step(model):
                        dict_out[model] = model_val
        out = {self.clock: dict_out}
        return out

    @staticmethod
    def dict_res_to_array(res_dict: Dict, var_name: str, order='var-time'):
        """
        Will convert the dictionary output of `run_multistep_to_dict` into
        an array. Due to the potential for heterogeneous shapes across variables
        only one variable can be extracted.
        :param res_dict:
        :param order:
                if  order=='var-time' then all the timesteps of each variable
                will be next to each other. For example:
                var1_time1, var1_time2,..., var1_timeT, var2_time1.....
                if  order=='time-var' then all the variables for each timestep
                will be next to each other. For example:
                var1_time1, var2_time1,..., varN_time1, var1_time2.....
        :param force_vectors:
                if any of the variables are multidimensional, they will
                broken into vectors and given
        :param var_name:
        :return:
        """
        assert order in ('var-time', 'time-var')
        out = []
        for t in res_dict:
            if var_name in res_dict[t]:
                val = res_dict[t][var_name].T
                cols = val.shape[1]
                if len(val.shape) == 1:
                    val = val.reshape(-1, 1)
                out.append(val)
        if order == 'var-time':
            out = np.stack(out, axis=-1)
            # this will create an array (sims, sub_var, time)
            out = out.reshape((out.shape[0], -1), order='C')
            # this will create an array (sims, varsXtimes)
        else:
            out = np.concatenate(out, axis=-1)
            # this will create an array (sims, timesXvars)
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
    return Fraction(1, int(round(1/flt, 0)))

ESG = Model