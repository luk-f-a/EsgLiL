# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 08:17:50 2018

@author: luk-f-a
"""
import os
import sys
parent = os.path.dirname
sys.path.append(parent(parent(parent(__file__))))

from esglil.common import Variable
from esglil.esg import Model
from esglil.equity_models import GeometricBrownianMotion
from esglil.rng import IndWienerIncr, MCMVIndWienerIncr
from esglil.multithreaded_rng import MultithreadedRNG
import numexpr as ne



def value(x, input_name):
    if input_name == 'self_1':
        return x.value_t
    else:
        #print(input_name, type(getattr(x, input_name)), isinstance(getattr(x, input_name), Variable) )
        if isinstance(getattr(x, input_name), Variable):
            return getattr(x, input_name).value_t
        else:
            return getattr(x, input_name)

def test1():
    dW = IndWienerIncr(1, 10_000_000)
    S = GeometricBrownianMotion(mu=1, sigma=0.2, dW=dW)
    esg = Model(dt_sim=1, dW=dW, S=S)
    out = esg.run_multistep_to_pandas(dt_out=1, max_t=20)
    
#    print(out)
    
def test2():
    ne.set_num_threads(2)
    dW = IndWienerIncr(1, 1000000)
    S = GeometricBrownianMotion(mu=1, sigma=0.2, dW=dW)
    esg = Model(dt_sim=1, dW=dW, S=S)
    esg.initialize()
    delta_t = 1
    for t in range(1,41):
        for eq_name in esg.eq:
            eq = esg.eq[eq_name]
            if hasattr(eq, 'ne_eq'):
                fc = eq.ne_eq
                local_dict = {}
                for inp_name in eq.ne_inputs:
                    if hasattr(eq, inp_name):
                        local_dict[inp_name] = value(eq, inp_name)
                    if inp_name=='self_1':
                        local_dict[inp_name] = eq.value_t
                for inp in fc.input_names:
                    if not inp in local_dict:
                        local_dict[inp] = locals()[inp]
                args = ne.necompiler.getArguments(fc.input_names, local_dict=local_dict)
#                print(type(eq), eq.value_t)
#                print(eq_name, t, args)
                fc(*args, out=eq.value_t, order='K', casting='safe', ex_uses_vml=False)
            else:
                eq.run_step(t)
                 
#    print(esg.df_value_t(out_vars=None))
    
def test3():
    ne.set_num_threads(4)
    dW = IndWienerIncr(1, 1000000, generator='mc-numpy')
    S = GeometricBrownianMotion(mu=1, sigma=0.2, dW=dW)
    esg = Model(dt_sim=1, dW=dW, S=S)
    d = esg.run_multistep_to_dict(dt_out=1, max_t=20, use_numexpr=True)
    
def test4():
    ne.set_num_threads(4)
    dW = MCMVIndWienerIncr(1, 10_000_000,1, 0,1,mcmv_time=1, generator='mc-multithreaded',
                           n_jobs=4)
    S = GeometricBrownianMotion(mu=1, sigma=0.2, dW=dW)
    esg = Model(dt_sim=1, dW=dW, S=S)
    d = esg.run_multistep_to_dict(dt_out=1, max_t=20, use_numexpr=True)    
    
#    print(d[2]['S'])
    
import datetime
tic = datetime.datetime.now()
test1()
print(datetime.datetime.now()-tic)
tic = datetime.datetime.now()
test4()
print(datetime.datetime.now()-tic)
