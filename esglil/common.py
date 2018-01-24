#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:16:19 2017

@author: luk-f-a
"""

class ValueDict(dict):
    def __getitem__(self, key):
        if isinstance(key, float) or len(key)==1:
            val = dict.__getitem__(self, key)
        else:
            val = dict.__getitem__(self, key[0])
        return val
    
class Variable(object):
    __slots__ = []
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if inputs[1] is self:
            return ufunc(inputs[0], inputs[1].value_t)
        else:
            return ufunc(inputs[0].value_t, inputs[1])
    
    def __add__(self, other):
        if isinstance(other, Variable):
            return self.value_t + other.value_t
        else:
            return self.value_t + other
    
    def __sub__(self, other):
        if isinstance(other, Variable):
            return self.value_t - other.value_t
        else:
            return self.value_t - other
        
    def __mul__(self, other):
        return self.value_t * other
    
    def __matmul__(self, other):
        if isinstance(other, Variable):
            return self.value_t @ other.value_t
        else:
            return self.value_t @ other
        
    def __truediv__(self, other):
        return self.value_t / other
        
    def __pow__(self, other):
        return self.value_t ** other        
    
    def __radd__(self, other):
        if isinstance(other, Variable):
            return other.value_t + self.value_t
        else:
            return other + self.value_t
        
    def __rsub__(self, other):
        if isinstance(other, Variable):
            return other.value_t - self.value_t
        else:
            return other - self.value_t

    def __rtruediv__(self, other):
        if isinstance(other, Variable):
            return other.value_t / self.value_t
        else:
            return other / self.value_t        
        
    def __rmul__(self, other):
        if isinstance(other, Variable):
            return other.value_t * self.value_t
        else:
            return other * self.value_t
    
    def __rmatmul__(self, other):
        if isinstance(other, Variable):
            return other.value_t @ self.value_t
        else:
            return other @ self.value_t
    
    def __getitem__(self, key):
        return VariableView(self, key)
    
    def __call__(self):
        """ideally should only be used for debugging
        """
        return self.value_t
    
    @property
    def shape(self):
        return self.value_t.shape

        
class VariableView(object):
    __slots__ = ['variable', 'key']

    def __init__(self, var, key):
        self.variable = var
        self.key = key
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if inputs[1] is self:
            return ufunc(inputs[0], self.variable.value_t[self.key,...])
        else:
            return ufunc(self.variable.value_t[self.key,...], inputs[1])
    
    def __add__(self, other):
        if isinstance(other, Variable):
            return self.variable.value_t[self.key,...] + other.value_t
        else:
            return self.variable.value_t[self.key,...] + other
    
    def __sub__(self, other):
        if isinstance(other, Variable):
            return self.variable.value_t[self.key,...] - other.value_t
        else:
            return self.variable.value_t[self.key,...] - other
        
    def __mul__(self, other):
        return self.variable.value_t[self.key,...] * other        
    
    def __truediv__(self, other):
        return self.variable.value_t[self.key,...] / other
        
    def __pow__(self, other):
        return self.variable.value_t[self.key,...] ** other        
    
    def __radd__(self, other):
        if isinstance(other, Variable):
            return other.value_t + self.variable.value_t[self.key,...]
        else:
            return other + self.variable.value_t[self.key,...]
        
    def __rsub__(self, other):
        if isinstance(other, Variable):
            return other.value_t - self.variable.value_t[self.key,...]
        else:
            return other - self.variable.value_t[self.key,...]

    def __rtruediv__(self, other):
        if isinstance(other, Variable):
            return other.value_t / self.variable.value_t[self.key,...]
        else:
            return other / self.variable.value_t[self.key,...]
        
    def __rmul__(self, other):
        if isinstance(other, Variable):
            return other.value_t * self.variable.value_t[self.key,...]
        else:
            return other * self.variable.value_t[self.key,...]
    
    def __iter__(self):
        return iter(self.variable.value_t[self.key,...])
    
    def __call__(self):
        """ideally should only be used for debugging
        """
        return self.variable.value_t[self.key,...]
    
    def _replace_variable(self, old_object, new_object):
        if self.variable is old_object:
            self.variable = new_object
        
    @property
    def shape(self):
        if hasattr(self.variable, 'shape'):
            return self.variable.shape[1:]
        else:
            return None
    
        
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
        
    def _replace_links(self, name, old_object, new_object):
        for item in self.__slots__:
            if not hasattr(self, item):
                continue
            obj = getattr(self, item)
            if obj is old_object:
                setattr(self, item, new_object)
            elif isinstance(obj, VariableView):
                obj._replace_variable(old_object, new_object)

                
    def __call__(self):
        return self.value_t
    
class TimeDependentParameter(Variable):
    __slots__ = ('f', 'value_t')
     
    def __init__(self, function):
        self.f = function 
        self.value_t = self.f(0)
    
    def run_step(self, t):
        self.value_t = self.f(t)
        
    def __call__(self, t):
        return self.f(t)
    
    def _replace_links(self, name, old_object, new_object):
        pass

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