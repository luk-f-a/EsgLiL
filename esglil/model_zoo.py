#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:39:15 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import TimeDependentParameter, SDE
from esglil.ir_models import HullWhite1fShortRate, HullWhite1fCashAccount,HullWhite1fBondPrice, hw1f_B_function
from esglil.esg import ESG
from esglil.equity_models import GeometricBrownianMotion
from esglil import rng

def Esg_e_sr_bonds_cash(delta_t, sims, rho, bond_prices, hw_a = 0.001, hw_sigma = 0.01,
                           gbm_sigma=0.2):
    assert type(bond_prices) is dict
#    assert len(bond_prices.shape)==2
#    assert bond_prices.shape[1]==1
    delta_t = delta_t
    rho = 0.2
    corr = [[1, rho], [rho, 1]]
    C = np.diag([delta_t, delta_t])
    cov = C*corr*C.T 
    dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=cov)
    B_fc = hw1f_B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma)
    B = TimeDependentParameter(B_fc)
    r = HullWhite1fShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    
    P_0 = np.array(list(bond_prices.values())).reshape(-1,1)
    T = np.array(list(bond_prices.keys())).reshape(-1,1)
    P = HullWhite1fBondPrice(B=B, a=hw_a, r=r, sigma=hw_sigma, dW=dW[0], 
                             P_0=P_0, T=T)
    C = HullWhite1fCashAccount(r=r)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg = ESG(dt_sim=delta_t, dW=dW,B=B, r=r, cash=C, P=P, S=S)
    
    return esg
