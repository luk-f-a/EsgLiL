#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:39:15 2017

@author: luk-f-a
"""
import numpy as np
from esglil.common import TimeDependentParameter, FunctionOfVariable
from esglil.ir_models import hw1f_euler as hw1feuler
from esglil.ir_models import hw1f_annual_exact as hw1fexact
from esglil.ir_models import simple_annual_model as simple_annual

from esglil.ir_models.common import DeterministicBankAccount
from esglil.esg import ESG
from esglil.equity_models import (GeometricBrownianMotion, GBM_exact,
                                  EquitySimpleAnnualModel, EquityExcessReturns)
from esglil import rng
from typing import Dict

def esg_e_sr_bonds_cash(delta_t, sims, rho, bond_prices, hw_a = 0.001, 
                        hw_sigma = 0.01, gbm_sigma=0.2, seed=None):
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
    ind_dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=indep_cov,
                           seed=seed)
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    B_fc, f, p0 = hw1feuler.B_function_dict(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
    B = TimeDependentParameter(B_fc)
    r = hw1feuler.ShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    
    T = np.array(list(bond_prices.keys())).reshape(-1,1)
    P = hw1feuler.BondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                             P_0=p0, f=f, T=T)

    C = hw1feuler.CashAccount(r=r)
    W = rng.WienerProcess(dW)
    ind_W = rng.WienerProcess(ind_dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, ind_W=ind_W,
              B=B, r=r, cash=C, P=P, S=S)
    
    return esg


def esg_e_sr_bonds_cash_2levels(delta_t_l1, delta_t_l2, sims, rho, bond_prices, hw_a = 0.001, hw_sigma = 0.01,
                           gbm_sigma=0.2, const_tau=None, seed=None):
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
    ind_dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=indep_cov,
                           seed=seed)
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    B_fc, f, p0 = hw1feuler.B_function(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
    B = TimeDependentParameter(B_fc)
    r = hw1feuler.ShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    
    if const_tau is not None:
        constP = hw1feuler.ConstantMaturityBondPrice(a=hw_a, r=r, sigma=hw_sigma,  
                                 P_0=p0, f=f, tau=const_tau.reshape(-1,1))
    C = hw1feuler.CashAccount(r=r)
    W = rng.WienerProcess(dW)
    ind_W = rng.WienerProcess(ind_dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg_l1 = ESG(dt_sim=delta_t_l1, ind_dW=ind_dW, dW=dW, W=W, ind_W=ind_W,
              B=B, r=r, cash=C,  S=S)
    esg_l2 = ESG(dt_sim=delta_t_l2, esg_l1=esg_l1, constP=constP)
    
    return esg_l2


def get_gbm_hw_nobonds(delta_t, sims,  rho=None, bond_prices=None,
                       hw_a = 0.001, hw_sigma = 0.01, gbm_sigma=0.2, 
                       custom_ind_dw=None, seed=None):
    """Returns an esg model with short rate, cash and equity  
    using a GBM and HW models.
    
    delta_t: float or Fraction
        time delta to use for euler simulation
      
    sims: int
        amount of simulations to run - this is the amount of "flat" sims 
        or outer sims in a MCMV scheme
    
    inner_sims: int
        amount of inner simulations after time 1
        
    rho: float 
        correlation between HW and GBM Brownian motions.
    
    bond_prices: dictionary {maturity: price}
        prices for notional 1 zero-coupon bonds, ie discount factors.
        
    custom_ind_dw: Rng object
        Use this Rng instead of the default one
        
    """
    if custom_ind_dw:
        assert custom_ind_dw.sims==sims
        ind_dW = custom_ind_dw
    else:
        ind_dW = rng.IndWienerIncr(dims=2, sims=sims, mean=0, delta_t=delta_t,
                                   generator='mc-numpy', seed=seed)
        
    corr = np.array([[1, rho], [rho, 1]])
    C = np.diag([np.sqrt(delta_t), np.sqrt(delta_t)])
    dep_cov = C@corr@C.T 
    indep_cov = np.diag([delta_t, delta_t])
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    B_fc = hw1feuler.B_function_dict(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=False)
    B = TimeDependentParameter(B_fc)
    r = hw1feuler.ShortRate(B=B, a=hw_a, sigma=hw_sigma, dW=dW[0])
    C = hw1feuler.CashAccount(r=r)
    W = rng.WienerProcess(dW)
    ind_W = rng.WienerProcess(ind_dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, ind_W=ind_W,
              B=B, r=r, cash=C,  S=S)
    return esg


def get_gbm_hw_2levels(delta_t_l1, delta_t_l2, sims, rho, bond_prices, 
                       hw_a = 0.001, hw_sigma = 0.01,
                       gbm_sigma=0.2, const_tau=None, out_bonds=None,
                       custom_ind_dw=None, seed=None):
    
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
        
    custom_ind_dw: Rng object
        Use this Rng instead of the default one
    
    """
    assert type(bond_prices) is dict
    B_fc, f, p0 = hw1feuler.B_function_dict(bond_prices=bond_prices, a=hw_a, sigma=hw_sigma,
                           return_p_and_f=True)
    esg_l1 = get_gbm_hw_nobonds(delta_t_l1, sims, rho, bond_prices, 
                       hw_a, hw_sigma, gbm_sigma=gbm_sigma, 
                       custom_ind_dw=custom_ind_dw, seed=seed)
     
    opt_kwargs = {}
    if const_tau is not None:
        hw_const_bond = hw1feuler.ConstantMaturityBondPrice
        constP = hw_const_bond(a=hw_a, r=esg_l1['r'], sigma=hw_sigma,  
                               P_0=p0, f=f, tau=const_tau.reshape(-1,1))
        opt_kwargs['constP'] = constP
        
    if out_bonds is not None:
        T = out_bonds
        P = hw1feuler.BondPrice(a=hw_a, r=esg_l1['r'], sigma=hw_sigma, 
                                 P_0=p0, f=f, T=T)
        opt_kwargs['P'] = P
 
    esg_l2 = ESG(dt_sim=delta_t_l2, esg_l1=esg_l1, **opt_kwargs)
    return esg_l2


def get_hw_gbm_annual(sims, rho, bond_prices: Dict[float, float],
                      hw_alpha=0.5, hw_sigma=0.01, gbm_sigma=0.2,
                      const_tau=None, out_bonds=None,
                      real_estate=False, rho_re=None, gbm_re_sigma=0.05,
                      custom_ind_z=None, seed=None):
    """Returns an esg model loop with short_rate cash, equity,
    bonds and/or constant maturity bonds using a GBM and HW models on an
    annual simulation step.
        
        
    sims: int
        amount of simulations to run - this is the amount of "flat" sims 
        or outer sims in a MCMV scheme
    
    rho: float 
        correlation between HW and GBM Brownian motions.


    bond_prices: dictionary {maturity: price}
       prices for notional 1 zero-coupon bonds, ie discount factors.
        
    const_tau: 1-d array
        constant time to maturity to track bond prices (instead of calculating
        all prices at every step)
        
    out_bonds: 1-d array
        maturity of bonds to track at every time step

    rho_re: float
        correlation between HW and GBM (real estate) Brownian motions.
        
    custom_ind_z: Rng object
        Use this Rng instead of the default one

    :param real_estate:
    :return: ESG with following variables:  ind_Z, Z_r_c, Z_r_e,
               cash, r, S,constP and P
    
    """
    dims = 4 if real_estate else 3
    if real_estate:
        assert rho_re is not None and 0 <= rho_re <= 1
    assert 0 <= rho <= 1
    if custom_ind_z:
        assert custom_ind_z.sims == sims
        ind_Z = custom_ind_z
    else:
        indep_cov = np.diag([1] * dims)
        mean_vec = [0] * dims
        cov_mat = indep_cov
        ind_Z = rng.NormalRng(dims=dims, sims=sims, mean=mean_vec,
                              cov=cov_mat, seed=seed)

    Z_r_c  = hw1fexact.StochasticDriver(ind_Z[[0, 1]], hw_sigma, hw_alpha)
    Z_r_e = rng.CorrelatedRV(ind_Z[[0, 2]], input_cov=np.diag([1, 1]),
                            target_cov=np.array([[1, rho], [rho, 1]]))
    b_vec = hw1fexact.calibrate_b_function(bond_prices, hw_alpha, hw_sigma)
    b = lambda t: b_vec[int(t)]
    g = hw1fexact.get_g_function(b_s=b, alpha=hw_alpha)
    r_zero = hw1fexact.calc_r_zero(bond_prices[1])
    r = hw1fexact.ShortRate(g=g, sigma_hw=hw_sigma, alpha_hw=hw_alpha, 
                          r_zero=r_zero, Z=Z_r_c[0])
    h = hw1fexact.get_h_function(b=b, alpha=hw_alpha)
    cash = hw1fexact.CashAccount(h=h, sigma_hw=hw_sigma,  alpha_hw=hw_alpha, 
                               r=r, Z=Z_r_c[1])
    S = EquityExcessReturns(cash=cash, sigma=gbm_sigma, Z=Z_r_e[1])
    opt_kwargs = {}
    if real_estate:
        Z_r_re = rng.CorrelatedRV(ind_Z[[0, 3]], input_cov=np.diag([1, 1]),
                                 target_cov=np.array([[1, rho_re], [rho_re, 1]]))
        RE = EquityExcessReturns(cash=cash, sigma=gbm_re_sigma, Z=Z_r_re[1])
        opt_kwargs['Z_r_re'] = Z_r_re
        opt_kwargs['RE'] = RE
    if const_tau is not None:
        hw_const_bond = hw1fexact.ConstantMaturityBondPrice
        constP = hw_const_bond(alpha=hw_alpha, r=r, sigma=hw_sigma,  
                               P_0=bond_prices, tau=const_tau.reshape(-1,1),
                               h=h)
        opt_kwargs['constP'] = constP
        
    if out_bonds is not None:
        T = out_bonds
        for t in T:
            P = hw1fexact.BondPrice(alpha=hw_alpha, r=r, sigma_hw=hw_sigma,
                                 h=h, T=t)
            opt_kwargs['P{}'.format(t)] = P
        
    esg =  ESG(dt_sim=1, ind_Z = ind_Z, Z_r_c=Z_r_c, Z_r_e=Z_r_e,
               cash=cash, r=r, S=S, **opt_kwargs)
    return esg
    
    

def esg_equity(delta_t, sims, r=0.03, gbm_sigma=0.2, generator='mc-numpy',
                 dask_chunks=1, seed=None, n_threads=1):
    dW = rng.IndWienerIncr(dims=1, sims=sims, mean=0, delta_t=delta_t,
                           generator=generator, dask_chunks=dask_chunks, 
                           seed=seed, n_threads=n_threads)
    W = rng.WienerProcess(dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW)
    esg = ESG(dt_sim=delta_t, dW=dW, W=W, S=S)
    
    return esg



def esg_equity2(delta_t, sims, r=0.03, gbm_sigma=0.2, seed=None):
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
    ind_dW = rng.NormalRng(dims=2, sims=sims, mean=[0, 0], cov=indep_cov,
                           seed=seed)
    dW = rng.CorrelatedRV(rng=ind_dW, input_cov=indep_cov , target_cov=dep_cov)
    dW1d = rng.NormalRng(dims=1, sims=sims, mean=[0], cov=np.array([[float(delta_t)]]))
    W = rng.WienerProcess(dW)
    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW[1])
    cash = DeterministicBankAccount(r)
    esg = ESG(dt_sim=delta_t, ind_dW=ind_dW, dW=dW, W=W, dW1d=dW1d, S=S, cash=cash)
    return esg


def esg_equity_exact(delta_t, sims, r=0.03, gbm_sigma=0.2, seed=None):
    dW = rng.NormalRng(dims=1, sims=sims, mean=[0], seed=seed,
                       cov=np.array([[float(delta_t)]]))
    W = rng.WienerProcess(dW)
#    S = GeometricBrownianMotion(mu=r, sigma=gbm_sigma, dW=dW)
    S = GBM_exact(mu=r, sigma=gbm_sigma, W=W)
    esg = ESG(dt_sim=delta_t, dW=dW, W=W, S=S)
    
    return esg


def esg_simple_annual_model(sims, generator, custom_z=None, seed=None):
    
    if custom_z:
        assert custom_z.sims==sims
        ind_z = custom_z
    else:
        ind_z = rng.IndWienerIncr(dims=2, sims=sims, mean=0, delta_t=1,
                                   generator=generator, seed=seed)
        
    b=0.1
    a=0.1
    r_zero = 0.02
    sigma_e = 0.2
    sigma_r = 0.002
    rho = 0.5
    corr = np.array([[1, rho], [rho, 1]])
    C = np.diag([1, 1])
    dep_cov = C@corr@C.T 
    indep_cov = np.diag([1, 1])
    N = rng.CorrelatedRV(rng=ind_z, input_cov=indep_cov , target_cov=dep_cov)
    
    r = simple_annual.ShortRate(b=b, a=a, sigma=sigma_r, r_zero=r_zero,
                                   N=N[0])
    C = simple_annual.CashAccount(r=r)
    S = EquitySimpleAnnualModel(r=r, sigma=sigma_e, N=N[1], s_zero=100)
    esg = ESG(dt_sim=1, ind_Z=ind_z, N=N, r=r, cash=C,  S=S)
    return esg
    