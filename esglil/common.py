#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:16:19 2017

@author: luk-f-a
"""

class Variable(object):
    def __add__(self, other):
        return self.value_t + other
    
    def __sub__(self, other):
        return self.value_t - other
        
    def __mul__(self, other):
        return self.value_t * other        
    
    def __truediv__(self, other):
        return self.value_t / other
        
    def __pow__(self, other):
        return self.value_t ** other        
    
    def __radd__(self, other):
        return self.value_t + other
        
    def __rsub__(self, other):
        return other - self.value_t
        
    def __rmul__(self, other):
        return self.value_t * other  
    
    
def _check_interface(obj):
    for attr in Variable:
        assert hasattr(obj, attr) 
        
class SDE(Variable):
    __slots__ = ('value_t', 't_1')
    
    def _check_valid_params(self):
        for param in self.__slots__:
            if param[0] != '_' and param != 'value_t':
                assert _check_interface(self.__getattribute__(param))

    def run_step(self, t):
        """Implement here the code to provide the next iteration of the
        equation
        """
        raise NotImplementedError
        
#    def __call__(self):
#        return self.value_t
    
class TimeDependentParameter(Variable):
    __slots__ = ('f')
     
    def __init__(self, function):
        self.f = function 
        self.value_t = self.f(0)
    
    def run_step(self, t):
        self.value_t = self.f(t)
        
#    def __call__(self):
#        return self.out

#class ConstantParameter(Variable):
#    __slots__ = ('out')
#    def __init__(self, value):
#        self.value_t = value
#    
#    def run_step(self):
#        pass
#        
#    def __call__(self):
#        return self.out