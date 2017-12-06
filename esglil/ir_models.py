#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:53:54 2017

@author: luk-f-a
"""
import xarray as xr
import numpy as np
from scipy.interpolate import make_interp_spline
from .common import SDE

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
        if type(bond_prices) is xr.DataArray:
            prices = list(bond_prices.values())
            time_shifts = list(bond_prices.coords['maturity'].values())
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
    r_zero = -(np.ln(p_tp1)-np.ln(pclass HullWhite1fShortRate(object):
    """class for (1 Factor) Hull White model of short interest rate
    This class only implements the short rate
    SDE: dr(t)= b(t)+dy(t)
         where y(t)=-a*r(t)+sigma*dW(t)
         

    The integral of b(t) between 0 and T is denoted as B(t) and can be used
    instead of b(t). B is deterministic and y(t) is stochastic. In practice,
    the simulations are based on r(t)= B(t)+y(t)
    All other parameters retain their meaning from the standard
    model parametrization dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)
    All parameters must be scaled to the unit of time used for the timestep.
    For example, if the steps are daily (1/252 of a year) then daily a, B and 
    sigma must be provided. This is because the class does not take a delta_t
    parameter, and therefore it must be embedded in the parameters such that
    dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t) => dr(t)=[b(t)-a*r(t)]+sigma*dW(t)
    
     Parameters
    ----------
        
    B(t) : callable (with t as only argument)
        integral from 0 to t of b(t). 
        
    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation
        
    """    
    __slots__ = ('B', 'a', 'sigma', '_yt', 'r')
    
    def __init__(self, B=None, a=None, sigma=None, dW=None):
    
        self.B = B
        self.a = a
        self.sigma = sigma
        self.y_t = 0
        self.dW = dW

    def _check_valid_params(self):
        assert  (self.a is not None and
        self.B is not None and self.sigma is not None)

    def _to_callable(self, param):
        if not callable(param):
            if type(param) is float:
                param = lambda t: param
            else:
                raise TypeError
        return param
    
    def _process_params(self):
        if self.B is not None:
            self.B = self._to_callable(self.B)
        if self.a is not None:
            self.a = self._to_callable(self.a)
        if self.sigma is not None:
            self.sigma = self._to_callable(self.sigma)

    def run_step(self):
        self._yt += -self.a()*self._yt+self.sigma()*self.dW()
        self.r = self.B() + self._yt
        
    def __call__(self):
        return self.r_t))/tp1
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
    return xr.DataArray(list(dict_.values()), dims=['maturity'],
                                 coords=[list(dict_.keys())])

def hw1f_B_function(bond_prices, a, sigma):
    """Based on zero coupon bond prices, it will return a function B(t) 
    which is the integral between 0 and T of parameter b(t) in 
    the hw1f model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)    
    
    Parameters
    ----------
    bond_prices: dictionary or DataArray
        initial bond prices to be moved forward in time. If passing a 
        dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values

    Returns
    ----------
    B(t) : function
        integral of b(s) between 0 and t
    
    r_zero: scalar
        initial value of the short rate. Formally f(0,0) where r(t, T) is the
        instantaneous forward rate at time t for a maturity of T
        
    """
    

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
    return B, f(0)

class HullWhite1fShortRate(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class only implements the short rate
    SDE: dr(t)= b(t)+dy(t)
         where y(t)=-a*r(t)+sigma*dW(t)
         

    The integral of b(t) between 0 and T is denoted as B(t) and can be used
    instead of b(t). B is deterministic and y(t) is stochastic. In practice,
    the simulations are based on r(t)= B(t)+y(t)
    All other parameters retain their meaning from the standard
    model parametrization dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)
    All parameters must be scaled to the unit of time used for the timestep.
    For example, if the steps are daily (1/252 of a year) then daily a, B and 
    sigma must be provided. This is because the class does not take a delta_t
    parameter, and therefore it must be embedded in the parameters such that
    dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t) => dr(t)=[b(t)-a*r(t)]+sigma*dW(t)
    
     Parameters
    ----------
        
    B(t) : callable (with t as only argument)
        integral from 0 to t of b(t). 
        
    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation
        
    """    
    __slots__ = ('B', 'a', 'sigma', '_yt', 'dW')
    
    def __init__(self, B=None, a=None, sigma=None, dW=None):
        self.B = B
        self.a = a
        self.sigma = sigma
        self._yt = 0
        self.dW = dW
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
#        self._yt += -self.a()*self._yt+self.sigma()*self.dW()
#        self.out = self.B() + self._yt
        self._yt += -self.a*self._yt*(t-self.t_1)+self.sigma*self.dW
        self.value_t = self.B + self._yt        
        self.t_1 = t
    
    

class HullWhite1fBondPrice(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the bond prices for maturity T
    SDE: dP(t, T)/P(t,T) = r(t)*dt + sigma/a*(1-exp(-a*(T-t)))*dW
         
    for the Hull White model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)

   
     Parameters
    ----------

    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation

    T: scalar or numpy array
        Bond maturity        
    """    
    __slots__ = ('T', 'a', 'sigma', 'P_0', 'r', 'dW', 'dt')
    
    def __init__(self, B=None, a=None, sigma=None, dW=None):
        self.T = T
        self.a = a
        self.sigma = sigma
        self.dW = dW
        self.value_t= P_0
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.value_t * (1 + self.r*(t-self.t_1)
                                 +self.sigma/self.a*(1-np.exp(-self.a*(self.T-t)))*self.dW)      
        self.t_1 = t
        
    
class HullWhite1fCashAccount(SDE):
    """class for (1 Factor) Hull White model of short interest rate
    This class implements the cash account
    SDE: dC(t)/C(t) = r(t)*dt
         
    for the Hull White model dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)

   
     Parameters
    ----------

    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation

    T: scalar or numpy array
        Bond maturity        
    """    
    __slots__ = ('T', 'a', 'sigma', 'P_0', 'r', 'dW', 'dt')
    
    def __init__(self, r=None):
        self.r = r
        self.value_t = 1
        self.t_1 = 0
        #self._check_valid_params()

    def run_step(self, t):
        self.value_t = self.value_t * exp(self.r*(t-self.t_1))
        self.t_1 = t

        
class HullWhite1fModel(object):
    """class for (1 Factor) Hull White model of short interest rate
    This class can output short rate, cash index and bond prices
    SDE: dr(t)=[b(t)-a*r(t)]dt+sigma*dW(t)
    The integral of b(t) between 0 and T is denoted as B(t) and can be used
    instead of b(t) since r(t) = B(t)+y(t), where B is deterministic and 
    y(t)=-a*r(t)+sigma*dW(t)
    
     Parameters
    ----------
    b(t) : DataArray or callable (with t as only argument)
        b parameter in SDE. Only one of b or B can be supplied, not both.
        
    B(t) : DataArray or callable (with t as only argument)
        integral from 0 to t of b(t). Only one of b or B can be supplied, not both.
        
    a: scalar
        mean reversion speed
    
    sigma : scalar
        standard deviation
        
    r_zero : scalar
        initial value of short rate.
        
    bond_prices: dictionary or DataArray
        initial bond prices to be moved forward in time. If passing a 
        dictionary it should be {maturity: price}, if passing a DataArray
        it should contain a dimension 'maturity' with the prices as values
        
    delta_t_in, delta_t_out: scalars
        
    """
    
    __slots__ = ('b', 'B', 'a', 'sigma', 'p_t', 'y_t', 'cash_t', 'delta_t_in', 'delta_t_out', 'r',
                 '_use_xr')
    
    def __init__(self, b=None, B=None, a=None, sigma=None, r_zero=1, 
                 bond_prices=None, delta_t_out=1, delta_t_in=1):
    
        self.b = b
        self.B = B
        self.a = a
        self.sigma = sigma
        self.delta_t_in = delta_t_in
        self.delta_t_out = delta_t_out
        self.r = r_zero
        self._use_xr = None
        self.p_t = bond_prices
        self.y_t = 0
        self.cash_t = 0
        
    def _check_valid_params(self):
        assert  (self.a is not None and
        ((self.b is None)!=(self.B is None)) and
          self.sigma is not None)
        assert self.delta_t_out >= self.delta_t_in
        dt_ratio = round(self.delta_t_out/self.delta_t_in,5)
        assert dt_ratio == int(dt_ratio), ('delta out is {} and'
        ' delta in is {}. Ratio is {}').format(self.delta_t_out, self.delta_t_in,
                      round(self.delta_t_out / self.delta_t_in,5))

    def _check_valid_X(self, X):
        #TODO: check that X is either an xarray with the right dimensions
        # or a numpy array with the right dimensions (time dimension should be of size 1)
        if self._use_xr is None:
            if type(X) is xr.DataArray:
                self._use_xr = True
            else:
                self._use_xr = False
        else:
            if (type(X) is xr.DataArray) != (self._use_xr):
                raise TypeError("Generate called with and without xarray input")
        pass

    def _to_callable(self, param):
        if type(param) is xr.DataArray:
            param = lambda t: param.loc[{'timestep':t}]
        else:
            if not callable(param):
                raise TypeError        
        return param
    
    def _process_params(self):
        if self.b is not None:
            self.b = self._to_callable(self.b)
        if self.B is not None:
            self.B = self._to_callable(self.B)
        if type(self.p_t) is dict:
            self.p_t = _bond_dict_to_xr(self.p_t)        
        
    def transform(self, X):
        """
        Produces simulations of 1 factor Hull White model short rate
        
        Parameters
        ----------       
        X array of Brownian motion increments (dW) simulations
        
        
        Returns
        -------
        Short rate simulations calculated as r(t)=r(t-1)+Δr(t)
                                        Δr(t)=[b(t)-a*r(t)]dt+sigma*X
        If bond_prices were provided initially, then they are simulated as
         ΔB(t)=B(t)*[r(t)]dt+sigma/a*(1-exp(-a(T-t)))*X
         and provided as separate svars
                                        
        
        """

        a = self.a
        b = self.b
        B = self.B
        sigma = self.sigma 
        self._process_params()
        self._check_valid_params()
        self._check_valid_X(X)
        dt_ratio = int(round(self.delta_t_out/self.delta_t_in,5))
        if self._use_xr:
            r = xr.DataArray(np.empty(X.shape), dims=X.dims, coords=X.coords)
            r.coords['svar']=['short_rate']
            cash_ix = xr.DataArray(np.empty(X.shape), dims=X.dims, coords=X.coords)
            cash_ix.coords['svar']=['cash_index']
            if self.p_t is not None: 
                n_bonds = len(self.p_t)
                empty_p = np.repeat(np.empty(X.shape), n_bonds, axis=0)
                bond_coords = ['bond_{}'.format(float(i)) 
                              for i in self.p_t.coords['maturity']]
                coords = dict(X.coords)
                coords['svar'] = bond_coords
                p = xr.DataArray(empty_p, dims=X.dims, coords=coords)

            if b is not None:    
                r_t = self.r
            else:
                y_t = self.y_t
            cash = self.cash_t
            p_t = self.p_t
            
            for ts in r.timestep:
                t = ts*self.delta_t_in
                dW = X.loc[{'timestep':ts}]
                if b is not None:
                    b_t = b(t)
                    r_t += (b_t-a*r_t)*self.delta_t_in+sigma*dW
                else:
                    y_t += -a*y_t*self.delta_t_in+sigma*dW
                    B_t = B(t)
                    r_t = B_t + y_t
                r.loc[{'timestep':ts}] = r_t
                #cash
                cash = cash+r_t*self.delta_t_in
                cash_ix.loc[{'timestep':ts}] = np.exp(cash)
                #if bond prices must be calcualted, then move them forward
                if self.p_t is not None: 
                    T = self.p_t.coords['maturity']
                    #print(p_t)
                    p_t = p_t + p_t*(r_t*self.delta_t_in+
                                     sigma/a*(1-np.exp(-a*(T-t)))*dW)
                    p.loc[{'timestep':ts}] = p_t.squeeze()
            self.r = r_t
            self.p_t = p_t
            self.y_t = y_t
            self.cash_t = cash
            r_out = r[{'timestep':slice(dt_ratio-1, None, dt_ratio)}]
            cash_out = cash_ix[{'timestep':slice(dt_ratio-1, None, dt_ratio)}]
            if self.p_t is not None: 
                p_out = p[{'timestep':slice(dt_ratio-1, None, dt_ratio)}]
                out = xr.concat([r_out, cash_out, p_out], dim='svar')
            else:
                out = xr.concat([r_out, cash_out], dim='svar')
            out.attrs['delta_t'] = self.delta_t_out
        else:
            self.r += (b_t-a*r_t)*self.delta_t_in+sigma*X.loc[{'timestep':t}]
            out = self.r
        return out