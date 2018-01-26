#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:39:15 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import TimeDependentParameter, SDE
from esglil.ir_models import (HullWhite1fShortRate, HullWhite1fCashAccount, 
                              HullWhite1fBondPrice, hw1f_B_function,
                              HullWhite1fConstantMaturityBondPrice)
from esglil.ir_models import DeterministicBankAccount
from esglil.esg import ESG
from esglil.equity_models import GeometricBrownianMotion, GBM_exact
from esglil import rng

def esg_e_sr_bonds_cash(delta_t, sims, rho, bond_prices, hw_a = 0.001, 
                        hw_sigma = 0.01, gbm_sigma=0.2):
    """Returns and esg model with equity, bonds and cash, driven by a gbm
    and a hw-1f model respectively
    """
    
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
    ind_W = rng.WienerProcess(ind_dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, ind_W=ind_W,
              B=B, r=r, cash=C, P=P, S=S)
    
    return esg

def esg_e_sr_bonds_cash_2levels(delta_t_l1, delta_t_l2, sims, rho, bond_prices, hw_a = 0.001, hw_sigma = 0.01,
                           gbm_sigma=0.2, const_tau=None):
    """
    const_tau: constant time to maturity to track bond prices
    """
    assert type(bond_prices) is dict
#    assert len(bond_prices.shape)==2
#    assert bond_prices.shape[1]==1
    corr = [[1, rho], [rho, 1]]
    C = np.diag([np.sqrt(delta_t_l1), np.sqrt(delta_t_l1)])
    dep_cov = C@corr@C.T 
    indep_cov = np.diag([delta_t_l1, delta_t_l1])
    ind_dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=indep_cov)
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    B_fc, f, p0 = hw1f_B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
    B = TimeDependentParameter(B_fc)
    r = HullWhite1fShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    
    if const_tau is not None:
        constP = HullWhite1fConstantMaturityBondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                                 P_0=p0, f=f, tau=const_tau.reshape(-1,1))
    C = HullWhite1fCashAccount(r=r)
    W = rng.WienerProcess(dW)
    ind_W = rng.WienerProcess(ind_dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg_l1 = ESG(dt_sim=delta_t_l1, ind_dW=ind_dW, dW=dW, W=W, ind_W=ind_W,
              B=B, r=r, cash=C,  S=S)
    esg_l2 = ESG(dt_sim=delta_t_l2, esg_l1=esg_l1, constP=constP)
    
    return esg_l2

def get_gbm_hw_nobonds(delta_t, sims, rho, bond_prices, 
                       hw_a = 0.001, hw_sigma = 0.01, gbm_sigma=0.2, 
                       custom_ind_dw=None,
                       use_dask=False, dask_chunks=1):
    """Returns an esg model with short rate, cash and equity  
    using a GBM and HW models.
    
    delta_t: float or Fraction
        time delta to use for euler simulation
      
    sims: int
        amount of simulations to run
    
    rho: float 
        correlation between HW and GBM Brownian motions.
    
    bond_prices: dictionary {maturity: price}
        prices for notional 1 zero-coupon bonds, ie discount factors.
        
    use_dask: whether to use dask instead of numpy
    
    use_cores: if using dask, how many cores (chunks) should be used.
    """
    if custom_ind_dw:
        ind_dW = custom_ind_dw
    else:
        if use_dask:
            gen = 'mc-dask'
        else:
            gen = 'mc-numpy'
                
        ind_dW = rng.IndWienerIncr(dims=2, sims=sims, mean=0, delta_t=delta_t,
                                   generator=gen, dask_chunks=dask_chunks)
    corr = [[1, rho], [rho, 1]]
    C = np.diag([np.sqrt(delta_t), np.sqrt(delta_t)])
    dep_cov = C@corr@C.T 
    indep_cov = np.diag([delta_t, delta_t])
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    B_fc = hw1f_B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=False)
    B = TimeDependentParameter(B_fc)
    r = HullWhite1fShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    C = HullWhite1fCashAccount(r=r)
    W = rng.WienerProcess(dW)
    ind_W = rng.WienerProcess(ind_dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, ind_W=ind_W,
              B=B, r=r, cash=C,  S=S)
    return esg
    
def get_gbm_hw_2levels(delta_t_l1, delta_t_l2, sims, rho, bond_prices, 
                       hw_a = 0.001, hw_sigma = 0.01,
                       gbm_sigma=0.2, const_tau=None, out_bonds=None,
                       use_dask=False, use_cores=1):
    """Returns an esg model loop with short_rate cash, equity on a more granular
    simulation level and a second level loop with a different delta_t for
    bonds and/or constant maturity bonds using a GBM and HW models.
    The first level is built using get_gbm_hw_nobonds
    
    delta_t_l1: float or Fraction
        time delta to use for models that require euler simulation. This
        parameter is passed to get_gbm_hw_nobonds
    
    delta_t_l2: float or Fraction
        time delta to use for HW bonds (which use a closed form price)
        
    sims, rho, bond_prices: see get_gbm_hw_nobonds documentation.
        
    const_tau: 1-d array
        constant time to maturity to track bond prices (instead of calculating
        all prices at every step)
        
    out_bonds: 1-d array
        maturity of bonds to track at every time step
        
    use_dask: whether to use dask instead of numpy
    
    use_cores: if using dask, how many cores (chunks) should be used.
    """
    assert type(bond_prices) is dict
    B_fc, f, p0 = hw1f_B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
    esg_l1 = get_gbm_hw_nobonds(delta_t_l1, sims, rho, bond_prices, 
                       hw_a, hw_sigma, gbm_sigma, None, use_dask, use_cores)
    opt_kwargs = {}
    if const_tau is not None:
        hw_const_bond = HullWhite1fConstantMaturityBondPrice
        constP = hw_const_bond(a=hw_a, r=esg_l1['r'], sigma=hw_sigma,  
                               P_0=p0, f=f, tau=const_tau.reshape(-1,1))
        opt_kwargs['constP'] = constP
        
    if out_bonds is not None:
        T = out_bonds
        P = HullWhite1fBondPrice(a=hw_a, r=esg_l1['r'], sigma=hw_sigma, 
                                 P_0=p0, f=f, T=T)
        opt_kwargs['P'] = P
 
    esg_l2 = ESG(dt_sim=delta_t_l2, esg_l1=esg_l1, **opt_kwargs)
    return esg_l2

def esg_equity(delta_t, sims, r=0.03, gbm_sigma=0.2, use_dask=False):
    dW = rng.IndWienerIncr(dims=1, sims=sims, mean=0, delta_t=delta_t,
                                   use_dask=use_dask, use_cores=1)
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


