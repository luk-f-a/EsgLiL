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

class ModelLoop(object):
    __slots = ['eq', 'clock', 'dt_sim']    
    def __init__(self, dt_sim, **models):
        self.eq = models
        self.clock = 0
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

    def __getitem__(self, key):
        return self.eq[key]

    
class Model(ModelLoop):

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
    
        
    def run_multistep_to_pandas(self, dt_out, max_t, out_vars=None):
        import pandas as pd
        dt_out = float_to_fraction(dt_out)
        assert max_t > float(self.clock)
        if dt_out < self.dt_sim:
            assert self.dt_sim % dt_out == 0
            return self._dense_run_to_pandas(dt_out, max_t, out_vars)
        else:
            assert dt_out % self.dt_sim == 0
            return self._sparse_run_to_pandas(dt_out, max_t, out_vars)

    
    def _sparse_run_to_pandas(self, dt_out, max_t, out_vars=None):
        """This method makes a run blah blah
        """
        out = []
        nb_steps = int(round(float((max_t-self.clock)/self.dt_sim),0))
        for ts in range(1, nb_steps+1):
            self.run_step()
            if ts % float(dt_out/self.dt_sim) == 0:
                out.append(self.df_value_t(out_vars))

        df = pd.concat(out, axis=0)
        return df
    
    def _dense_run_to_pandas(self, dt_out, max_t, out_vars=None):
        """
        This method allows for a dt_out which is smaller than the dt of the 
        model, thus allowing to record the full history of sub-models running on
        a more granular timegrid
        """
        out = []
        nb_steps = int(round(max_t/dt_out,0))
        for ts in range(1, nb_steps+1):
            t = ts*dt_out
            if t % self.dt_sim == 0:
                self.run_step()
                out.append(self.df_value_t(out_vars))
            else:
                for model in self.eq:
                    if hasattr(self.eq[model], 'run_multistep_to_pandas'):
                        mstep_fc = self.eq[model].run_multistep_to_pandas
                        m_df = mstep_fc(dt_out, max_t=t, out_vars=out_vars)
                        m_df = m_df.rename(index={orig:model+'_'+orig 
                                             for orig in m_df.index.get_level_values('model')})
                        out.append(m_df)
                        
        df = pd.concat(out, axis=0)
        return df

    def df_value_t(self, out_vars):
        import pandas as pd
        value_dict = {}
        df_list = []
        for model in self.eq:
            if hasattr(model, 'df_value_t'):
                val_df = model.df_value_t(out_vars)
                df_list.append(val_df)
                #maybe add here model name as prefix to the columns names 
            else:
                model_val = self.eq[model].value_t
                if isinstance(model_val, dict):
                    for sub_output in model_val:
                        if (out_vars is None) or (sub_output in out_vars):
                            val = model_val[sub_output]
                            key_name = model+'_'+sub_output
                            v_d = value_to_dictionary(key_name, val)
                            value_dict.update(v_d)
                else:
                    if (out_vars is None) or (model in out_vars):
                        value_dict.update(value_to_dictionary(model, model_val))
        df = pd.DataFrame.from_dict(value_dict).assign(time=self.clock)
        df.index.names = ['sim']
        df.columns.names = ['model']
        df = df.set_index(['time'], append=True)
        df = df.unstack('time')
        df = pd.concat([df]+df_list, axis=1)
        
        #df = df.stack()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df
        
    
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