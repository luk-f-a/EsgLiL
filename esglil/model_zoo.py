#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:39:15 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import TimeDependentParameter, SDE
from esglil.ir_models import HullWhite1fShortRate, HullWhite1fCashAccount,HullWhite1fBondPrice, hw1f_B_function
from esglil.ir_models import DeterministicBankAccount
from esglil.esg import ESG
from esglil.equity_models import GeometricBrownianMotion, GBM_exact
from esglil import rng

def esg_e_sr_bonds_cash(delta_t, sims, rho, bond_prices, hw_a = 0.001, hw_sigma = 0.01,
                           gbm_sigma=0.2):
    assert type(bond_prices) is dict
#    assert len(bond_prices.shape)==2
#    assert bond_prices.shape[1]==1
    corr = [[1, rho], [rho, 1]]
    C = np.diag([np.sqrt(delta_t), np.sqrt(delta_t)])
    dep_cov = C@corr@C.T 
    indep_cov = np.diag([delta_t, delta_t])
    ind_dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=indep_cov)
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    B_fc, f, p0 = hw1f_B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
    B = TimeDependentParameter(B_fc)
    r = HullWhite1fShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    
    T = np.array(list(bond_prices.keys())).reshape(-1,1)
    P = HullWhite1fBondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                             P_0=p0, f=f, T=T)
    C = HullWhite1fCashAccount(r=r)
    W = rng.WienerProcess(dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, B=B, r=r, cash=C, P=P, S=S)
    
    return esg

def esg_e_sr_bonds_cash_test(delta_t, sims, rho, bond_prices, hw_a = 0.001, hw_sigma = 0.01,
                           gbm_sigma=0.2):
    assert type(bond_prices) is dict
#    assert len(bond_prices.shape)==2
#    assert bond_prices.shape[1]==1
    corr = [[1, rho], [rho, 1]]
    C = np.diag([np.sqrt(delta_t), np.sqrt(delta_t)])
    dep_cov = C@corr@C.T 
    indep_cov = np.diag([delta_t, delta_t])
    ind_dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=indep_cov)
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    B_fc, f, p0 = hw1f_B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
    B = TimeDependentParameter(B_fc)
    r = HullWhite1fShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    
    T = np.array(list(bond_prices.keys())).reshape(-1,1)
    P = HullWhite1fBondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                             P_0=p0, f=f, T=T)
    C = HullWhite1fCashAccount(r=r)
    W = rng.WienerProcess(dW)
    #S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    S = GeometricBrownianMotion(mu=0, sigma=gbm_sigma, dW=ind_dW[1])
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, B=B, r=r, cash=C, P=P, S=S)
    
    return esg

def esg_equity(delta_t, sims, r=0.03, gbm_sigma=0.2):
    dW = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[float(delta_t)]])
    W = rng.WienerProcess(dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, W=W, S=S)
    
    return esg

def esg_equity2(delta_t, sims, r=0.03, gbm_sigma=0.2):
    """covar calculations
    target_cov = L.L'  (L is cholesky)
    X: independent dW
    Z: dependednt dW
    to obtain E[ZZ']= target_cov then
    Z=MX where M=L@[[sigma^-1, 0],[0, sigma^-1]]
    and sigma=sqrt(delta_t)
    """
    corr = [[1, 0.2], [0.2, 1]]
    C = np.diag([np.sqrt(delta_t), np.sqrt(delta_t)])
    dep_cov = C@corr@C.T 
    indep_cov = np.diag([delta_t, delta_t])
    ind_dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=indep_cov)
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    dW1d = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[float(delta_t)]])
    W = rng.WienerProcess(dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    cash = DeterministicBankAccount(r)
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, dW1d=dW1d, S=S, cash=cash)
    return esg



def esg_equity_exact(delta_t, sims, r=0.03, gbm_sigma=0.2):
    dW = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=[[float(delta_t)]])
    W = rng.WienerProcess(dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW)
    Se = GBM_exact(mu=r, sigma=gbm_sigma, W=W)
    esg = ESG(dt_sim=delta_t,  dW=dW, W=W, Se=Se, S=S)
    
    return esg


