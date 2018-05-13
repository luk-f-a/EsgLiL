#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This model is an exact simulation of Hull White 1 factor model for annual
timesteps.

All formulas are based on Glasserman's book:
    Monte Carlo Methods in Financial Engineering
    
    
Created on Sun May 13 14:00:49 2018
@author: luk-f-a
"""
import numpy as np
from esglil.common import StochasticVariable, ValueDict
from esglil.rng import CorrelatedRV

class StochasticDriver(StochasticVariable):
    """class to generate the correlated normal variables
    needed for exact yearly simulation of short rate and cash account
    
    dr(t)=alpha*[b(t)-r(t)]dt+sigma*dW(t)
    
    Parameters
    ----------
    
    rng: random number generator
        multivariate standard normal variables, with 2 independent dimensions
        
    sigma_hw: float
        hull white volatility paramenter
        
    alpha_hw: float
        hull white alpha mean reversion speed parameter 
    """
    def __new__(cls, rng, sigma_hw, alpha_hw):
        sigma_r = (sigma_hw**2/2/alpha_hw*(1-np.exp(-2*alpha_hw)))**0.5
        sigma_y = (sigma_hw**2/alpha_hw**2*
                       (1+1/2/alpha_hw*(1-np.exp(-2*alpha_hw))
                        +2/alpha_hw*(np.exp(-alpha_hw)-1)))**0.5
        sigma_r_y = (sigma_hw**2/2/alpha_hw*
                       (1+np.exp(-2*alpha_hw)
                        -2*np.exp(-alpha_hw)))
#        print('sigma_r_y','sigma_r','sigma_y', sigma_r_y,sigma_r,sigma_y)
        rho_r_y = sigma_r_y/sigma_r/sigma_y
#        print(rho_r_y)
        return CorrelatedRV(rng, input_cov=np.diag([1,1]), 
                            target_cov=np.array([[1,rho_r_y], [rho_r_y,1]]))
        
        
class ShortRate(StochasticVariable):
    """class for (1 Factor) Hull White model of short interest rate
    This class only implements the short rate
    
    dr(t)=alpha*[b(t)-r(t)]dt+sigma*dW(t)
    according to Glasserman's book formulae

    
     Parameters
    ----------
        
    g : callable (with two arguments)
        integral of exp(-alpha(T-t))b(t)
        
    alpha: scalar or Variable object
        mean reversion speed
    
    sigma_hw : scalar
        standard deviation
        
    Z: random number generator
        1-d standard normal variable
    """    

    
    __slots__ = ['sigma_r', 'mu_r', 'alpha', 'Z']
    def __init__(self, g, sigma_hw, alpha_hw, r_zero, Z):
        self.mu_r = lambda t:alpha_hw*g(t)
        self.sigma_r = np.sqrt(sigma_hw**2/2/alpha_hw*(1-np.exp(-2*alpha_hw)))
        self.alpha = alpha_hw
        self.value_t = r_zero
        self.Z = Z
        self.t_1 = 0
        
    def run_step(self, t):
        self.value_t = (np.exp(-self.alpha)*self.value_t+self.mu_r(self.t_1)
                        +self.sigma_r*self.Z)
        self.t_1 = t
        
class CashAccount(StochasticVariable):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the cash account
    
    dr(t)=alpha*[b(t)-r(t)]dt+sigma*dW(t)
    according to Glasserman's book formulae

    
     Parameters
    ----------
        
    h(t, T) : callable (with 2 arguments)
        double integral of exp(-alpha(T-s)*b(s))
        
    alpha: scalar or Variable object
        mean reversion speed
    
    sigma_hw : scalar
        standard deviation
        
    Z: random number generator
        1-d standard normal variable, must be correlated to the Z for the 
        short rate in specific way, given by the HWyearlyStochasticDriver class
    """ 
    
    __slots__ = ['sigma_y', 'mu_y', 'alpha', 'Y_t', 'Z', 'r']
    
    def __init__(self, h, sigma_hw, alpha_hw, r, Z):
        self.mu_y = lambda t: (1/alpha_hw*(1-np.exp(-alpha_hw))*r
                               +alpha_hw*h(t))
        self.sigma_y = np.sqrt(sigma_hw**2/alpha_hw**2*
                       (1+1/2/alpha_hw*(1-np.exp(-2*alpha_hw))
                        +2/alpha_hw*(np.exp(-alpha_hw)-1)))
        self.value_t = 1
        self.Y_t = 0
        self.Z = Z
        self.r = r
        self.t_1 = 0
        
    def run_step(self, t):
        assert self.r.t_1==self.t_1, "Cash account simulation must be done before r simulation for the new period"
        self.Y_t = self.Y_t+self.mu_y(self.t_1)+self.sigma_y*self.Z
        self.value_t = np.exp(self.Y_t)
        self.t_1 = t
        
        
class BondPrice(StochasticVariable):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the bond prices for maturity T
    P(t, T) = exp[-A(t,T)*r(t) + C(t,T)]
         
    for the Hull White model dr(t)=alpha*[b(t)-r(t)]dt+sigma*dW(t)
    according to Glasserman's book formulae
    
    with 
    
    A(t,T)=1/alpha*(1-exp(-alpha(T-t)))
    
    C(t,T)=-alpha*H(t, T)+ sigma**2*alpha**2*[(T-t)+1/2/alpha*(1-exp(-2*alpha*(T-t)))+
                                             +2/alpha*(exp(-alpha(T-t))-1)]
   
     Parameters
    ----------

    alpha: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation

    T: scalar or array
        Bond maturity
        
    h(t, T): callable
        double integral of exp(-alpha(u-s))*b(s)
    """    
    __slots__ = ('T', 'alpha', 'sigma', 'r', 'h')
    
    def __init__(self, alpha, sigma_hw, r, h, T):
        self.T = T
        self.alpha = alpha
        self.sigma = sigma_hw
        self.r = r
        self.h = h
        self.t_1 = 0
        self.run_step(0)
#        self._check_valid_params()

        
    def run_step(self, t):
        alpha = self.alpha
        sigma = self.sigma
        r = self.r
        h = self.h
        T = self.T
        
        self.value_t = hw_bond_price(alpha, T, t, h, sigma, r, lib=np)
        self.t_1 = t

def hw_bond_price(alpha, T, t, h, sigma, r, lib=np):
        exp = lib.exp
        A = 1/alpha*(1-exp(-alpha*(T-t)))
        C = (-alpha*h(t, T)+ sigma**2/2/alpha**2*((T-t)+1/2/alpha*(1-exp(-2*alpha*(T-t)))+
                                             +2/alpha*(exp(-alpha*(T-t))-1)))
        return exp(-A*r+C)

     
class ConstantMaturityBondPrice(StochasticVariable):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the bond prices for time to maturity tau (instead of
    maturity T as the HullWhite1fBondPrice).
    The class is a more efficient (if slightly over-specialized) way of
    tracking a bond portfolio without having to calculate the price of every
    possible bond as the HullWhite1fBondPrice class does.
    
    P(t, T) = exp[-A(t,T)*r(t) + C(t,T)]

    tau = T - t
         
    for the Hull White model dr(t)=alpha*[b(t)-r(t)]dt+sigma*dW(t)
    according to Glasserman's book formulae
    
    with 
    
    A(t,T)=1/alpha*(1-exp(-alpha(T-t)))
    
    C(t,T)=-alpha*H(t, T)+ sigma**2*alpha**2*[(T-t)+1/2/alpha*(1-exp(-2*alpha*(T-t)))+
                                             +2/alpha*(exp(-alpha(T-t))-1)]
   
     Parameters
    ----------

    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation

    tau: scalar or array
        Bond maturity
        
   
    P_0: dict
        P[T], bond prices at time 0 observed (market) for maturity T
    
    r: stochastic variable
        short rate
    """    
    __slots__ = ('tau', 'A', 'alpha', 'sigma', 'r', 'h')
    
    def __init__(self, alpha, sigma, r, P_0, tau, h):
        self.tau = tau
        self.alpha = alpha
        self.sigma = sigma
        self.r = r
        if isinstance(tau, np.ndarray):
            self.h = np.vectorize(h, excluded='t')
        else:
            self.h = h
        self.value_t = ValueDict({float(t):P_0[t] for t in tau[:,0]})
        self.t_1 = 0
#        self._check_valid_params()

        
    def run_step(self, t):
        alpha = self.alpha
        sigma = self.sigma
        r = self.r
        h = self.h  
        T = self.tau + t

        bond_prices = hw_bond_price(alpha, T, t, h, sigma, r, lib=np)
        
        self.value_t = ValueDict({float(t):bond for t, bond 
                                    in zip(self.tau[:,0], bond_prices)})
        self.t_1 = t


def get_g_function(b_s, alpha):
    """Return g function based on b function and alpha paramter
    """
    k_m = (1-np.exp(-alpha))/alpha
    def g(t):
        return b_s(t)*k_m
    return g
        

def get_h_function(b, alpha):
    """Return h function based on b function and alpha paramter
    """
    k_p = (np.exp(alpha)-1)/alpha
    k_m = (1-np.exp(-alpha))/alpha
    m = (alpha+np.exp(-alpha)-1)/(alpha**2)
    def h(t, T=None):
        t = int(t)
        if T is None:
            T = t+1
        else:
            T = int(T)
        h_val = 0
        for j in range(t, T):
            for i in range(t, j):
#                print('int1', b(i)*np.exp(-alpha*(j-i)))
                h_val += b(i)*np.exp(-alpha*(j-i))
        h_val *= k_p*k_m
        for j in range(t, T):
            h_val += m*b(j)
#            print('m', m)
#            print('b{}'.format(j), b(j))
#            print('int2', m*b(j))
        return h_val
    return h

def calc_r_zero(b1_price):
    return -(np.log(b1_price))

def h_from_B(B_T, T, t, r, alpha, sigma):
    """returns the value of h(t,T) given b(t,T) and alpha
    B_T: float
        bond price for maturity T
    r: float
        r(t)
    t: float
        valuation time
    T: float
        maturity of bond
    alpha: float
        mean reversion speed
    sigma: float
        hw volatility
    """
    A = 1/alpha*(1-np.exp(-alpha*(T-t)))
    h = (-1/alpha*(np.log(B_T)+A*r)+sigma**2/2/alpha**3*((T-t)
                            +1/2/alpha*(1-np.exp(-2*alpha*(T-t)))
                            +2/alpha*(np.exp(-alpha*(T-t))-1)))
    return h

def b_from_h(t, T, h_T, h_T_1, b, alpha):
    """return b(T-1) (since b is shifted left)
    from h(t,T) and h(t,T-1)
    
    b: 1-d array
        b values upto T-2
    """
#    print(t, T, h_T, h_T_1)
    k_p = (np.exp(alpha)-1)/alpha
    k_m = (1-np.exp(-alpha))/alpha
    next_b = (h_T- h_T_1-k_p*k_m*sum([b[i]*np.exp(-alpha*(T-1-i)) for i in range(t,T-1)]))
    next_b = next_b * alpha**2 / (alpha+np.exp(-alpha)-1)
    return next_b

def calibrate_b_function(bond_prices:dict, alpha, sigma):
    p_0 = 1 #bond price at time t
    p_1 = bond_prices[1]
    r_zero = -(np.log(p_1)-np.log(p_0))
    h_T_1 = 0
    b = []
    for T in bond_prices:
        h = h_from_B(B_T=bond_prices[T], T=T, t=0, r=r_zero, alpha=alpha,
                     sigma=sigma)
        b_T_1 = b_from_h(t=0, T=T, h_T=h, h_T_1=h_T_1, b=b, alpha=alpha)
        b.append(b_T_1)
        h_T_1 = h
    return b
