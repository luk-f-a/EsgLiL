# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:30:24 2017

@author: CHK4436
"""
import xarray as xr
import numpy as np
import pandas as pd
from fractions import Fraction

class Rng(object):
    """Base class for random number generators
    """
    __slots__ = ('shape', 'out')

    def __init__(self, dims, sims):
        self.shape = (dims, sims)
                
       
    def generate(self):
        """Implement here the code to provide the next iteration of the
        random number generator
        """
        raise NotImplementedError
  
    def __call__(self):
        return self.out
    
class Variable(object):
    def __add__(self, other):
        return self.out + other
    
    def __sub__(self, other):
        return self.out - other
        
    def __mul__(self, other):
        return self.out * other        
    
    def __truediv__(self, other):
        return self.out / other
        
    def __pow__(self, other):
        return self.out ** other        
    
    def __radd__(self, other):
        return self.out + other
        
    def __rsub__(self, other):
        return other - self.out
        
    def __rmul__(self, other):
        return self.out * other    
    
class NormalRng(Rng, Variable):
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
    __slots__ = ('mean', 'cov')
                 
    def __init__(self,  dims, sims, mean, cov):
        Rng.__init__(self, dims, sims)
        self.mean = mean
        self.cov = cov
        
    def _check_valid_params(self):
        #TODO: if output is numpy, mean must be size 1 and cov 1x1
        # if output is xr size of mean and cov must agree with svar dim
        pass
    
    
    def run_step(self, t):
        self.out = self.generate()
        
    def generate(self):
        """Return the next iteration of the random number generator
        """
        self._check_valid_params()
        
        out = np.random.multivariate_normal(self.mean, self.cov, 
                                               self.shape,
                                               check_valid='raise')
        return out
    
class SDE(object):
    __slots__ = ('out')
    
    def _check_valid_params(self):
        for param in self.__slots__:
            if param[0] != '_' and param != 'out':
                assert callable(self.__getattribute__(param))

    def run_step(self, t):
        """Implement here the code to provide the next iteration of the
        equation
        """
        raise NotImplementedError
        
    def __call__(self):
        return self.out
        
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
        #self._check_valid_params()

    def run_step(self, t):
#        self._yt += -self.a()*self._yt+self.sigma()*self.dW()
#        self.out = self.B() + self._yt
        self._yt += -self.a*self._yt+self.sigma*self.dW
        self.out = self.B + self._yt        
        
    
    

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

    T: scalar
        Bond maturity        
    """    
    __slots__ = ('T', 'a', 'sigma', 'P_0', 'r', 'dW', 'dt')
    
    def __init__(self, B=None, a=None, sigma=None, dW=None):
        self.T = T
        self.a = a
        self.sigma = sigma
        self.dW = dW
        self.out= P_0
        self.dt = dt
        #self._check_valid_params()

    def run_step(self, t):
        self.out = self.out * (1 + self.r*self.dt 
                                 +self. sigma/self.a*(1-np.exp(-self.a*(self.T-t)))*self.dW)      
        
    
class ESG(object):
    def __init__(self, dt_sim, dt_out, **models):
        self.eq = models
        self.clock = 0
        self.dt_sim = Fraction(dt_sim)
        self.dt_out = Fraction(dt_out)
        assert float(self.dt_out/self.dt_sim) % 1 == 0
        
    def run_step(self):
        self.clock += self.dt_sim
        for model in self.eq:
            self.eq[model].run_step(float(self.clock))
            
    def full_run_to_pandas(self, max_t):
        out = {}
        for ts in range(1, int(float(max_t/self.dt_sim))+1):
            self.run_step()
            t = float(ts*self.dt_sim)
            if ts % float(self.dt_out/self.dt_sim) == 0:
                out[t] = pd.DataFrame(self.out)
                out[t].index.names = ['sim']
        return pd.concat(out, axis=0)
    
    @property
    def out(self):
        return {model: np.array(self.eq[model]()).squeeze() for model in self.eq}
#    def __call__(self):
#        return {model:self.eq[model]() for model in self.eq}
        
    
class TimeDependentParameter(Variable):
    __slots__ = ('f')
     
    def __init__(self, function):
        self.f = function 
        self.out = self.f(0)
    
    def run_step(self, t):
        self.out = self.f(t)
        
    def __call__(self):
        return self.out

#class ConstantParameter(object):
#    __slots__ = ('out')
#    def __init__(self, value):
#        self.value_t = value
#    
#    def run_step(self):
#        pass
#        
#    def __call__(self):
#        return self.out



def main():
    delta_t = 1/250
    max_t = 40
    dW = NormalRng(dims=1, sims=1000, mean=[0], cov=[[1/delta_t]])
#    a = ConstantParameter(0.2)
#    sigma = ConstantParameter(0.01)
    a = 0.001
    sigma = 0.01
    B = TimeDependentParameter(function=lambda t: 0.1)
    r = HullWhite1fShortRate(B=B, a=a, sigma=sigma, dW=dW)
    esg = ESG(dt_sim=delta_t, dt_out=1, dW=dW, B=B, r=r)
    df_full_run = esg.full_run_to_pandas(40)
    print(df_full_run)
    
main()
    