#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:18:45 2017

@author: small_screen
"""

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from esglil import rng
from esglil import equity_models
from esglil.esg import ESG
from sympy import symbols

class SymbolicdW(rng.Rng):
    """class for normal random number generation

     Parameters
    ----------
    dims: int
        Amount of dimensions in for the output normal variable
    
    sims : int
        Amount of simulations to produce in each timestep
        
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
        
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. 
        It must be symmetric and positive-semidefinite for proper sampling.
    """
    __slots__ = ('value_t')
                 
    def __init__(self,):
        pass
        
    def run_step(self, t):

        #self.value_t[...] = self.generate()
        self.value_t = symbols('dW_'+ str(t).replace('.','_'))
#        print(out.squeeze()[:2])
        return self.value_t
    
delta_t = 0.1
#dW = rng.IndependentWienerIncrements(dims=1, sims=5000, delta_t=delta_t, 
#                                     distributed=False)
dW = SymbolicdW()
S = equity_models.GeometricBrownianMotion(mu=0.02, sigma=0.2, dW=dW)
esg = ESG(dt_sim=delta_t, dW=dW, S=S)
#df_full_run = esg.run_multistep_to_pandas(dt_out=1, max_t=40)
for _ in range(3):
    esg.run_step()
    
out = esg.value_t['S']

