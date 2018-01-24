#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:46:28 2017

@author: lucio
"""


import os
import sys
sys.path.append('/home/lucio/mypyprojects/LiL/PyLiL/EsgLiL/')
from esglil.common import TimeDependentParameter, SDE
from esglil.ir_models import HullWhite1fShortRate, HullWhite1fCashAccount,HullWhite1fBondPrice
from esglil.esg import ESG
from esglil import rng
import esglil
import datetime
import numpy as np
import xarray as xr


class SwapRate(SDE):

    def __init__(self, *P):
        self.P = P
        self.value_t = 1
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        self.npv_sum = 0
        for bond_p in self.P:
            self.npv_sum = np.add(self.npv_sum, bond_p.value_t)
        self.npv_sum = np.add(self.npv_sum, - self.P[0].value_t)
        self.value_t = (self.P[0].value_t - self.P[len(self.P)-1].value_t)/self.npv_sum
        self.t_1 = t
        #0:00:03.377440
        
class SwapRate1(SDE):
    __slots__ = ['P']
    def __init__(self, P):
        self.P = P
        self.run_step(0)
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        npv_sum = sum(self.P[1:])
        self.value_t = (self.P[0] - self.P[-1])/npv_sum
        self.t_1 = t
        
class SwapRate2(SDE):
    def __init__(self, *P):
        self.P = P
        self.value_t = 1
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        self.npv_sum = 0
        for bond_p in self.P[1:]:
            self.npv_sum = np.add(self.npv_sum, bond_p.value_t)
        #self.value_t = (self.P[0].value_t - self.P[len(self.P)-1].value_t)/self.npv_sum
        self.value_t = (self.P[0].value_t - self.P[-1].value_t)/self.npv_sum
        self.t_1 = t

def main():
    bond_prices = {i: 1 for i in range(1,41)}
    a = 0.01
    sigma = 0.05
    delta_t = 1 #1/100
    dW = rng.NormalRng(dims=1, sims=10, mean=[0], cov=[[delta_t]])
    B = 0.03
    B = TimeDependentParameter(function=lambda t:0.03)
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    
    P = {'Bond_{}'.format(i):HullWhite1fBondPrice(B=B, a=a, r=r, sigma=sigma, dW=dW,
                             P_0=bond_prices[i], T=i) for i in range(1,41)}
    
    SR_5_5 = SwapRate1([P['Bond_{}'.format(i)] for i in range(5,11)])
    esg = ESG(dt_sim=delta_t, dW=dW, B=B, r=r, **P, SR=SR_5_5)
    df_sims=esg.full_run_to_pandas(dt_out=1, max_t=40)


def time_fc(fc, params):
    took = datetime.datetime.now()-datetime.datetime.now()
    for _ in range(5):
        now_ = datetime.datetime.now()
        fc(*params)
        took += datetime.datetime.now()-now_
    return took/5

print(time_fc(main, []))