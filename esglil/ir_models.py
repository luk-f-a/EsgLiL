#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:53:54 2017

@author: luk-f-a
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from esglil.common import SDE, ValueDict

from collections import Iterable

def hw1f_sigma_calibration(bond_prices, swaption_prices, a):
    """Based on zero coupon bond and swaption prices, it will return 
    a volatility parameter for the hw1f model 
    dr(t)=[b(t)-a*r(t)]dt+sigma dW(t)
    
     Parameters
    ----------
    bond_prices: dictionary or DataArray
        initial bond prices/discount factor, ie bond prices for a notional of 1.
        If passing a dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values
        
    swaption_vols: list of dictionaries
        initial swaption volatilities, ie bond prices for a notional of 1.
        The list of dictionary should look like
        [{'start': start1, 'tenor': tenor1, 'strike': strike1, 'normal_vol': market_vol1},
        ...
        {'start': startN, 'tenor', tenorN, 'strike':strikeN, 'normal_vol': market_volN}]
        If the original data is in a DataFrame, it can be quickly converted
        to this list of dictionaries format as  df.to_dict('records')
        provided that all data sits in data columns and not in the index. 
        Otherwise run df.reset_index().to_dict('records').
        For ATM strikes use None or 'ATM'.
    
    Example
    ---------
    bonds = {0: 1.0, 2:0.99, 100:0.5}
    swaptions = [{'start': 1, 'tenor': 5, 'strike': 0.01, 'normal_vol': 0.1148},
      {'start': 2, 'tenor': 4, 'strike': 0.01, 'normal_vol': 0.1108},
      {'start': 3, 'tenor': 3, 'strike': 0.01, 'normal_vol': 0.1070},
      {'start': 4, 'tenor': 2, 'strike': 0.01, 'normal_vol': 0.1021},
      {'start': 5, 'tenor': 10, 'strike': 0.01, 'normal_vol': 0.1000}]
    hw1f_sigma_calibration(bp, swp, 0.05)
    >>> 0.059216852521512736
    
    Returns
    ----------
    sigma: scalar
        diffusion parameter in HW1F model
    """
    import QuantLib as ql
    from collections import namedtuple
    #import math
    
    def create_swaption_helpers(data, index, term_structure, engine):
        swaptions = []
        fixed_leg_tenor = ql.Period(1, ql.Years)
        fixed_leg_daycounter = ql.Actual360()
        floating_leg_daycounter = ql.Actual360()
        nominal = 1.0
        vol_type = ql.Normal
        for d in data:
            strike = d['strike']
            if strike is None or strike=='ATM':
                strike = ql.nullDouble() 
            
            vol_handle = ql.QuoteHandle(ql.SimpleQuote(d['normal_vol']))
            assert(type(d['start']) is int)
            helper = ql.SwaptionHelper(ql.Period(d['start'], ql.Years),
                                       ql.Period(d['tenor'], ql.Years),
                                       vol_handle,
                                       index,
                                       fixed_leg_tenor,
                                       fixed_leg_daycounter,
                                       floating_leg_daycounter,
                                       term_structure,
                                       ql.CalibrationHelper.RelativePriceError ,
                                       strike,
                                       nominal,
                                       vol_type
                                       )
            helper.setPricingEngine(engine)
            swaptions.append(helper)
        return swaptions         
    
    def create_yield_term_structure(bond_prices):
        if type(bond_prices) is dict:
            prices = list(bond_prices.values())
            time_shifts = list(bond_prices.keys())
            sorted_list = sorted(zip(time_shifts, prices))
            time_shifts, prices = zip(*sorted_list)
            time_shifts = list(time_shifts)
            prices = list(prices)
#        if type(bond_prices) is xr.DataArray:
#            prices = list(bond_prices.values())
#            time_shifts = list(bond_prices.coords['maturity'].values())
        if time_shifts[0] != 0:
            time_shifts.insert(0, 0)
            prices.insert(0, 1.0)            
        #a reference date is needed because QL works based on a calendar
        # but it should not make a difference which date is "today"
        today = ql.Date(31, ql.December, 2016)
        day_count = ql.Thirty360()
        assert all([type(ts) is int for ts in time_shifts])
        dates = [today+ ql.Period(i, ql.Years) for i in time_shifts]
        curve = ql.DiscountCurve(dates, prices, day_count)
        term_structure = ql.YieldTermStructureHandle(curve)
        index = ql.Euribor1Y(term_structure)
        return term_structure, index

    #Core of the function below
    term_structure, index = create_yield_term_structure(bond_prices)
    constrained_model = ql.HullWhite(term_structure, a, 0.001);
    engine = ql.JamshidianSwaptionEngine(constrained_model)
    swaptions = create_swaption_helpers(swaption_prices, index,
                                        term_structure, engine)
    optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
    end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
    constrained_model.calibrate(swaptions, optimization_method, 
                                end_criteria, ql.NoConstraint(), [], 
                                [True, False])
    
    a_model, sigma = constrained_model.params()
    assert a_model==a
    return sigma

def hw1f_b_calibration_dict(bond_prices, a, sigma):
    """Based on zero coupon bond prices, it will return a function b(t) for
    the hw1f model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)
    
     Parameters
    ----------
    bond_prices: dictionary or DataArray
        initial bond prices to be moved forward in time. If passing a 
        dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values

    Returns
    ----------
    b(t) : function
        mean reversion level
    
    r_zero: scalar
        initial value of the short rate
    """
    import xarray as xr
    assert isinstance(bond_prices, dict)
    

    
    b = {}
    #short rate, t is current time, tp1 is "t plus 1", tp2 is t plus 2
    p_t = 1 #bond price at time t
    t = 0
    maturities = list(bond_prices.keys())
    tp1 = float(min(maturities))
    p_tp1 = bond_prices[tp1]
    r_zero = -(np.ln(p_tp1)-np.ln(p_t))/tp1
    t = tp1
    p_t = p_tp1
    #loop on all time steps where second derivatives can be calculated
    times = zip(maturities.values,
                maturities[1:].values,
                maturities[2:].values)
    for t, tp1, tp2 in times:
        assert int(t)==float(t), 'Temporarily, only integer maturities are allowed'
        t = float(t)
        p_t = bond_prices[t]
        p_tp1 = bond_prices[tp1]
        p_tp2 = bond_prices[tp2]
        f_t = -(np.ln(p_tp1)-np.ln(p_t))/(tp1 - t)
        f_tp1 = -(np.ln(p_tp2)-np.ln(p_tp1))/(tp2 - tp1)
        d_f = (f_tp1 - f_t)/(tp1 - t)
        b[t] = d_f + a*f_t + sigma**2/2/a*(1-np.exp(-2*a*t))

    b_fc = lambda t: b[t]
    return b_fc, r_zero
               
def hw1f_b_calibration(bond_prices, a, sigma):
    """Based on zero coupon bond prices, it will return a function b(t) for
    the hw1f model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)
    
     Parameters
    ----------
    bond_prices: dictionary or DataArray
        initial bond prices to be moved forward in time. If passing a 
        dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values

    Returns
    ----------
    b(t) : function
        mean reversion level
    
    r_zero: scalar
        initial value of the short rate
    """
    import xarray as xr
    if type(bond_prices) is dict:
        bond_prices = _bond_dict_to_xr(bond_prices)
    assert len(bond_prices.dims)==1
    assert 'maturity' in bond_prices.dims
    
    b = xr.DataArray(np.empty(len(bond_prices.maturity)), dims='time', 
                     coords=bond_prices.maturity)
    #short rate, t is current time, tp1 is "t plus 1", tp2 is t plus 2
    p_t = 1 #bond price at time t
    t = 0
    tp1 = float(bond_prices.coords['maturity'][0])
    p_tp1 = bond_prices[tp1]
    r_zero = -(np.ln(p_tp1)-np.ln(p_t))/tp1
    t = tp1
    p_t = p_tp1
    #loop on all time steps where second derivatives can be calculated
    times = zip(bond_prices.coords['maturity'].values,
                bond_prices.coords['maturity'][1:].values,
                bond_prices.coords['maturity'][2:].values)
    for t, tp1, tp2 in times:
        assert int(t)==float(t), 'Temporarily, only integer maturities are allowed'
        t = float(t)
        p_t = bond_prices[t]
        p_tp1 = bond_prices[tp1]
        p_tp2 = bond_prices[tp2]
        f_t = -(np.ln(p_tp1)-np.ln(p_t))/(tp1 - t)
        f_tp1 = -(np.ln(p_tp2)-np.ln(p_tp1))/(tp2 - tp1)
        d_f = (f_tp1 - f_t)/(tp1 - t)
        b[{'time':t}] = d_f + a*f_t + sigma**2/2/a*(1-np.exp(-2*a*t))

    b_fc = lambda t: b.loc[{'time':t}]
    return b_fc, r_zero

def _bond_dict_to_xr(dict_):
    import xarray as xr
    return xr.DataArray(list(dict_.values()), dims=['maturity'],
                                 coords=[list(dict_.keys())])

def hw1f_B_function(bond_prices, a, sigma, return_p_and_f=False):
    """Based on zero coupon bond prices, it will return a function B(t) 
    which is the integral between 0 and T of parameter b(t) in 
    the hw1f model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)    
    
    Parameters
    ----------
    bond_prices: dictionary or DataArray
        initial bond prices to be moved forward in time. If passing a 
        dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values
        
    a: scalar 
        mean reversion speed
    
    sigma : scalar
        standard deviation
    
    return_p_and_f: boolean
        whether the bond price function (interpolating the input data) 
        and the instantaneous forward rate function will be returned. These
        are both necessar in some formulas of HW bond prices

    Returns
    ----------
    B(t) : function
        integral of b(s) between 0 and t
    
    f(0,t): function
        instantaneous forward rate for maturity t, 
        
    """
    
    import xarray as xr
    if type(bond_prices) is dict:
        bond_prices = _bond_dict_to_xr(bond_prices)
    assert len(bond_prices.dims)==1
    assert 'maturity' in bond_prices.dims
    #calculate yield to maturity
    ytm_t = -np.log(bond_prices)/bond_prices.coords['maturity']
    abscisas = bond_prices.coords['maturity']
    ytm_0 = ytm_t[0]-abscisas[0]*(ytm_t[1]-ytm_t[0])/(abscisas[1]-abscisas[0])
    ytm_100 = ytm_t[-1]
    ytm_to_add = xr.DataArray(np.array([ytm_0, ytm_100]), 
                          dims='maturity', coords=[[0, 100]])
    ytm_t = ytm_t.combine_first(ytm_to_add)
    #abscisas = np.array([0] + list(abscisas)+[100])
    abscisas = ytm_t.coords['maturity']
    b_spline = make_interp_spline(abscisas, ytm_t)
    f = lambda t: b_spline(t)+t*b_spline.derivative()(t)
    B = lambda t: f(t)+ sigma**2/(2*a**2)*(1-np.exp(-a*t))**2
    if return_p_and_f:
        P = lambda t: np.exp(-t*b_spline(t))
        return B, f, P
    else:
        return B

def hw1f_B_function_dict(bond_prices, a, sigma, return_p_and_f=False):
    """Based on zero coupon bond prices, it will return a function B(t) 
    which is the integral between 0 and T of parameter b(t) in 
    the hw1f model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)    
    
    Parameters
    ----------
    bond_prices: dictionary or DataArray
        initial bond prices to be moved forward in time. If passing a 
        dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values
        
    a: scalar 
        mean reversion speed
    
    sigma : scalar
        standard deviation
    
    return_p_and_f: boolean
        whether the bond price function (interpolating the input data) 
        and the instantaneous forward rate function will be returned. These
        are both necessar in some formulas of HW bond prices

    Returns
    ----------
    B(t) : function
        integral of b(s) between 0 and t
    
    f(0,t): function
        instantaneous forward rate for maturity t, 
        
    """
    


    maturities = np.sort(np.array(list(bond_prices.keys())))
    prices = np.array(list(bond_prices.values()))
    #calculate yield to maturity
    ytm_t = -np.log(prices)/maturities
    abscisas = maturities
    ytm_0 = ytm_t[0]-abscisas[0]*(ytm_t[1]-ytm_t[0])/(abscisas[1]-abscisas[0])
    ytm_100 = ytm_t[-1]

    ytm_t = np.concatenate([np.array([ytm_0]), ytm_t, np.array([ytm_100])])
    #abscisas = np.array([0] + list(abscisas)+[100])
    abscisas = np.array([0]+maturities.tolist()+[100])
    b_spline = make_interp_spline(abscisas, ytm_t)
    f = lambda t: b_spline(t)+t*b_spline.derivative()(t)
    B = lambda t: f(t)+ sigma**2/(2*a**2)*(1-np.exp(-a*t))**2
    if return_p_and_f:
        P = lambda t: np.exp(-t*b_spline(t))
        return B, f, P
    else:
        return B
    
class HullWhite1fShortRate(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class only implements the short rate
    SDE: dr(t)= b(t)+dy(t)
         where dy(t)=-a*y(t)+sigma*dW(t)
         

    The integral of b(t) between 0 and T is denoted as B(t) and can be used
    instead of b(t). B is deterministic and y(t) is stochastic. In practice,
    the simulations are based on r(t)= B(t)+y(t)
    All other parameters retain their meaning from the standard
    model parametrization dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)
     
     Parameters
    ----------
        
    B(t) : callable (with t as only argument)
        integral from 0 to t of b(t). 
        
    a: scalar or SDE object
        mean reversion speed
    
    sigma : scalar
        standard deviation
        
    """    
    __slots__ = ('B', 'a', 'sigma', '_yt', 'dW')
    
    def __init__(self, B, a, sigma, dW):
        self.B = B
        self.a = a
        self.sigma = sigma
        self._yt = 0
        self.dW = dW
        self.t_1 = 0
        self.value_t = B #.value_t
        #self._check_valid_params()

    def run_step(self, t):
        self._yt = self._yt + self.dW*self.sigma-self.a*self._yt*(t-self.t_1)
        self.value_t = self.B + self._yt        
        self.t_1 = t

    def run_step_ne(self, t):
        self._evaluate_ne('_yt+dW*sigma-a*_yt*(t-t_1)',  
                          local_vars={'t': t}, out_var='_yt')
        self._evaluate_ne('B+_yt', out_var='value_t')
        self.t_1 = t    

class HullWhite1fBondPrice_WC(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the bond prices for maturity T
    SDE: dP(t, T)/P(t,T) = r(t)*dt + sigma/a*(1-exp(-a*(T-t)))*dW
        
    for the Hull White model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)

    using a weak convergence scheme. There is no path-convergence!!
   
     Parameters
    ----------

    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation

    T: scalar or numpy array
        Bond maturity        
    """    
    __slots__ = ('T', 'a', 'sigma', 'r', 'dW')
    
    def __init__(self, a, sigma, r, dW, P_0, T):
        self.T = T
        self.a = a
        self.sigma = sigma
        self.dW = dW
        self.r = r
        self.value_t = P_0
        self.t_1 = 0
        self._check_valid_params()

    def _check_valid_params(self):
        if isinstance(self.T, Iterable):
            assert isinstance(self.value_t, Iterable)
            assert len(self.T.shape) == len(self.dW.shape)+1, (
                    "T must have one more dimeneions than dW")
            assert len(self.value_t.shape) == len(self.dW.shape)+1
            assert len(self.value_t.shape) == len(self.value_t.shape)
        
    def run_step(self, t):
        self.value_t = self.value_t * (1 + self.r*(t-self.t_1)
                                 +self.sigma/self.a*(1-np.exp(-self.a*(self.T-t)))*(self.dW*1))      

        self.t_1 = t

    def run_step_ne(self, t):
        self._evaluate_ne('self_1*(1+r*(t-t_1)+sigma/a*(1-exp(-a*(T-t)))*dW',
                                   local_vars={'t': t}, out_var='value_t')
        self.t_1 = t
        
class HullWhite1fBondPrice(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the bond prices for maturity T
    P(t, T) = exp[-C(t,T)*r(t) + A(t,T)]
         
    for the Hull White model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)

    with 
    
    C(t,T)=1/a*(1-exp(-a(T-t)))
    
    A(t,T)=ln(Pm(0,T)/Pm(0,t))+fm(0,t)*C(t,T)-sigma^2/4a*(1-exp(-2at))*C(t,T)^2
   
     Parameters
    ----------

    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation

    T: scalar or array
        Bond maturity
        
    fm(0,t): function
        instantaneous forward rate at time 0 observed (market) and interpolated
    
    Pm(0,T): function
        bond prices at time 0 observed (market) and interpolated
    
    """    
    __slots__ = ('T', 'a', 'sigma', 'r', 'f', 'P_0')
    
    def __init__(self, a, sigma, r, P_0, f, T):
        self.T = T
        self.a = a
        self.sigma = sigma
        self.r = r
        self.value_t = P_0(T)
        self.P_0 = P_0
        self.f = f
        self.t_1 = 0
#        self._check_valid_params()

        
    def run_step(self, t):
        C = 1/self.a*(1-np.exp(-self.a*(self.T-t)))
        
        A = (np.log(self.P_0(self.T)/self.P_0(t))
            +self.f(t)*C
            -self.sigma**2/4/self.a*(1-np.exp(-2*self.a*t))*C**2)
        self.value_t = np.exp(-C*self.r+A)
        self.t_1 = t

    def run_step_ne(self, t):
        C = 1/self.a*(1-np.exp(-self.a*(self.T-t)))
        A = (np.log(self.P_0(self.T)/self.P_0(t))
            +self.f(t)*C
            -self.sigma**2/4/self.a*(1-np.exp(-2*self.a*t))*C**2)
        self._evaluate_ne('exp(-C*r+A)',
                          local_vars={'t': t,'C':C,'A':A},
                          out_var='value_t')

#        self._evaluate_ne('exp(-(1/a*(1-exp(-a*(T-t))))*r'
                           
#                               '+(log(P_0_T/P_0_t)+f_t*(1/a*(1-exp(-a*(T-t))))-'
#                               'sigma**2/4/a*(1-exp(-2*a*t))*(1/a*(1-exp(-a*(T-t))))**2))'
#                          local_vars={'t': t,'P_0_T':self.P_0(self.T),
#                                      'P_0_t':self.P_0(t), 'f_t': self.f(t)},
#                          out_var='value_t')
        self.t_1 = t
        
        
class HullWhite1fConstantMaturityBondPrice(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the bond prices for time to maturity tau (instead of
    maturity T as the HullWhite1fBondPrice).
    The class is a more efficient (if slightly over-specialized) way of
    tracking a bond portfolio without having to calculate the price of every
    possible bond as the HullWhite1fBondPrice class does.
    
    P(t, T) = exp[-C(t,T)*r(t) + A(t,T)]
    tau = T - t
         
    for the Hull White model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)

    with 
    
    C(t,T)=1/a*(1-exp(-a(T-t))) = 1/a*(1-exp(-a*tau)) = C(tau)
    
    A(t,T)=ln(Pm(0,T)/Pm(0,t))+fm(0,t)*C(t,T)-sigma^2/4a*(1-exp(-2at))*C(t,T)^2
   
     Parameters
    ----------

    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation

    T: scalar or array
        Bond maturity
        
    fm(0,t): function
        instantaneous forward rate at time 0 observed (market) and interpolated
    
    Pm(0,T): function
        bond prices at time 0 observed (market) and interpolated
    
    """    
    __slots__ = ('tau', 'C', 'a', 'sigma', 'r', 'f', 'P_0')
    
    def __init__(self, a, sigma, r, P_0, f, tau):
        self.tau = tau
        self.a = a
        self.sigma = sigma
        self.r = r
        self.value_t = ValueDict({float(t):P_0(t) for t in tau[:,0]})
        self.P_0 = P_0
        self.f = f
        self.t_1 = 0
        self.C = 1/self.a*(1-np.exp(-self.a*tau))
#        self._check_valid_params()

        
    def run_step(self, t):
        C = self.C
        A = (np.log(self.P_0(self.tau+t)/self.P_0(t))
            +self.f(t)*C
            -self.sigma**2/4/self.a*(1-np.exp(-2*self.a*t))*C**2)
        
        self.value_t = ValueDict({float(t):bond for t, bond 
                                    in zip(self.tau[:,0], np.exp(-C*self.r+A))})
        self.t_1 = t

    def run_step_ne(self, t):
        C = self.C
        A = self._evaluate_ne('(log(P_0_T/P_0_t)+f_t*C-'
                               'sigma**2/4/a*(1-exp(-2*a*t))*C**2)',  
                          local_vars={'t': t,'P_0_T':self.P_0(self.T),
                                      'P_0_t':self.P_0(self.t), 'f_t': self.f(t),
                                      'C':C})
        self.value_t = ValueDict({float(t):bond for t, bond 
                                    in zip(self.tau[:,0], np.exp(-C*self.r+A))})
        self.t_1 = t

        
class HullWhite1fCashAccount(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the cash account
    SDE: dC(t)/C(t) = r(t)*dt
         
    for the Hull White model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)

   
     Parameters
    ----------

    r: stochastic variable
        Hull-White short rate
         
    """    
    __slots__ = ('r')
    
    def __init__(self, r):
        self.r = r
        self.value_t = 1
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        #this version '*(1+rdt)' definitely works better than 
        #exponential capitalization '*exp(rdt)
        #with larger differences for smaller time deltas (which suggests
        #that it's the exponential which is wrong)
        self.value_t = self.value_t*(1+self.r*(t-self.t_1))
        self.t_1 = t

    def run_step_ne(self, t):
        self._evaluate_ne('self_1*(1+r*(t-t_1))',  
                          local_vars={'t': t}, out_var='value_t')
        self.t_1 = t
        
        
class DeterministicBankAccount(SDE):
    """for discounting under deterministic (constant) interest rates
    """
    __slots__ = ('r')
    
    def __init__(self, r):
        self.r = r
        self.value_t = 1
 
    def run_step(self, t):
        self.value_t = np.exp(self.r*t)       


class ShortRateSimpleAnnualModel(SDE):
    """class for a simple short rate model
    This class only implements the short rate
    SDE: r(t+1)= ar(t)+b(t)+sigma*N(i+1)
         
     
     Parameters
    ----------
        
    b(t) : scalar or SDE object
        
    a: scalar or SDE object
        
    sigma : scalar
        standard deviation
        
    r_zero: scalar
        initial value of the short rate
        
    N: stochastic variable or random number generator
        standar normal variable
        
        
    """    
    __slots__ = ('b', 'a', 'sigma', 'N')
    
    def __init__(self, b, a, sigma, r_zero, N):
        self.b = b
        self.a = a
        self.sigma = sigma
        self.N = N
        self.t_1 = 0
        self.value_t = r_zero #.value_t
        #self._check_valid_params()

    def run_step(self, t):
        assert t-self.t_1 == 1
        self.value_t = self.a*self.value_t+self.b+self.sigma*self.N
        self.t_1 = t

    def run_step_ne(self, t):
        self._evaluate_ne('a*self_1+b+sigma*N)',  out_var='value_t')
        self.t_1 = t    
        
class CashAccountSimpleAnnualModel(SDE):
    """class for the cash account under the simple annual model

   
     Parameters
    ----------
    r: stochastic varible
        short rate
    """    
    __slots__ = ('r')
    
    def __init__(self, r):
        self.r = r
        self.value_t = 1
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.value_t*np.exp(self.r*1)
        self.t_1 = t

    def run_step_ne(self, t):
        self._evaluate_ne('self_1*exp(r)',   out_var='value_t')
        self.t_1 = t