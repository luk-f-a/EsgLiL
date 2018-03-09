#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:16:19 2017

@author: luk-f-a
"""
try:
    import numexpr as ne
except ModuleNotFoundError:
    pass
except:
    raise

from functools import singledispatch
import operator

def value(x, input_name):
    """Matches input names with object x attributes to provide inputs
    to numexpr
    """
    if input_name == 'self_1':
        out = x.value_t
    else:
        out = get_value_of(getattr(x, input_name))
    return out

class ValueDict(dict):
    def __getitem__(self, key):
        if isinstance(key, float) or isinstance(key, str) or len(key)==1:
            val = dict.__getitem__(self, key)
        else:
            val = dict.__getitem__(self, key[0])
        return val
    
    def __setitem__(self, key):
        raise ValueError("Instead of setting a value, create a new object")
        
class Variable(object):
    """Base class for variables (deterministic or stochastic) in simulation
    It's basically a wrapper for an object holding the actual values, stored in
    "value_t". This object can in turn be an eager evaluation class
    (like a numpy array) or a lazy evaluation class (like a dask array).
    This variable class simply enables syntactic sugar like:
        bm = BrowianMotion(*params)
        gbm = np.exp(bm)

    ie, equations are classes on which mathematical functions operate on their
    current values

    """
    __slots__ = ['value_t']
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if inputs[1] is self:
            return ufunc(inputs[0], inputs[1].value_t)
        else:
            return ufunc(inputs[0].value_t, inputs[1])

    def _apply_l_op(self, other, op):
        if isinstance(other, Variable):
            return op(self.value_t, other.value_t)
        else:
            return op(self.value_t, other)

    def _apply_r_op(self, other, op):
        if isinstance(other, Variable):
            return op(other.value_t, self.value_t)
        else:
            return op(other, self.value_t)

    def __add__(self, other):
        return self._apply_l_op(other, operator.add)

    def __sub__(self, other):
        return self._apply_l_op(other, operator.sub)

    def __mul__(self, other):
        return self._apply_l_op(other, operator.mul)

    def __matmul__(self, other):
        return self._apply_l_op(other, operator.matmul)

    def __truediv__(self, other):
        return self._apply_l_op(other, operator.truediv)

    def __pow__(self, other):
        return self._apply_l_op(other, operator.pow)

    def __radd__(self, other):
        return self._apply_r_op(other, operator.add)

    def __rsub__(self, other):
        return self._apply_r_op(other, operator.sub)

    def __rtruediv__(self, other):
        return self._apply_r_op(other, operator.truediv)

    def __rmul__(self, other):
        return self._apply_r_op(other, operator.mul)

    def __rmatmul__(self, other):
        return self._apply_r_op(other, operator.matmul)

    def __rpow__(self, other):
        return self._apply_r_op(other, operator.pow)
    
    def __getitem__(self, key):
        return VariableView(self, key)

    def __call__(self):
        """ideally should only be used for debugging
        in other cases use function get_value_of
        """
        return self.value_t

    def get_value(self):
        return self.value_t

    @property
    def shape(self):
        return self.value_t.shape


class VariableView(Variable):
    __slots__ = ['variable', 'key']

    def __init__(self, var, key):
        self.variable = var
        self.key = key

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if inputs[1] is self:
            return ufunc(inputs[0], self.variable.value_t[self.key, ...])
        else:
            return ufunc(self.variable.value_t[self.key, ...], inputs[1])

    def _apply_l_op(self, other, op):
        if isinstance(other, Variable):
            return op(self.variable.value_t[self.key, ...], other.value_t)
        else:
            return op(self.variable.value_t[self.key, ...], other)

    def _apply_r_op(self, other, op):
        if isinstance(other, Variable):
            return op(other.value_t, self.variable.value_t[self.key, ...])
        else:
            return op(other, self.variable.value_t[self.key, ...])

    #All magic methods like __add__ are inherited from Variable
    
    def __iter__(self):
        return iter(self.variable.value_t[self.key,...])

    def __call__(self):
        """ideally should only be used for debugging
        in other cases use function get_value_of
        """
        return self.variable.value_t[self.key,...]

    def get_value(self):
        return self.variable.value_t[self.key,...]

    def _replace_variable(self, old_object, new_object):
        if self.variable is old_object:
            self.variable = new_object

    @property
    def value_t(self):
        return self.get_value()
    
    @property
    def shape(self):
        if hasattr(self.variable, 'shape'):
            return self.variable.shape[1:]
        else:
            return None    



@singledispatch
def get_value_of(arg):
    return arg

@get_value_of.register(Variable)
@get_value_of.register(VariableView)
def _(arg):
    return arg.get_value()


class StochasticVariable(Variable):
    """This class represents a stochastic variable. The randomness can be
    inherent (ie, it is a random number generator) or inherited
    (ie, it is a function of a stochastic variable).

    It provides an abstract method, run_step, that should be used
    to implement the value at time t, given the current value of all input
    variables

    Attribute t_1 stores the time at the previous iteration in case
    the equation defining the behaviour is a function of delta_time and not
    of time
    """
    __slots__ = ['t_1']

    def run_step(self, t):
        """Implement here the code to provide the next iteration of the
        equation
        """
        raise NotImplementedError

    def _replace_links(self, old_object, new_object):
        """ Replace the dependency to an object (old object) with a
        dependency to a new object.
        Useful to replace one node in an existing graph
        """
        for item in self.__slots__:
            if not hasattr(self, item):
                continue
            obj = getattr(self, item)
            if obj is old_object:
                setattr(self, item, new_object)
            elif isinstance(obj, VariableView):
                obj._replace_variable(old_object, new_object)

    def _inputs_to_dict(self):
        """ Collects the inputs to the equation that are stored as attributes
        of the object and puts them in a dictionary to be passed to
        numexpr
        """
        from itertools import chain
        slots = chain.from_iterable(getattr(cls, '__slots__', [])
                                    for cls in self.__class__.__mro__)
        slot_dict = {name: value(self, name) for name in slots if hasattr(self, name)}
        slot_dict.update({'self_1': self.value_t})
        return slot_dict

    def _evaluate_ne(self, ne_ex, local_vars=None, out_var=None):
        """this method evaluates the stochastic variable given its inputs
        using numexpr.
        It should only be called by the run_step_ne method.
        
        ne_ex: string
            formula to calculate. Any references to self or self.value_t
            should be expressed as self_1, to indicate it's the previous
            value of this formula
            
        local_vars: dict
            any variables needed for the calculation of ne_ex that cannot be
            found in the attributes of self, ie the object peforming the 
            calculation using NumExpr. One example is time, usually an input
            "t" to the run_step_ne method.
            
        out_var: string, optional
            name of the attribute of self where the calculations should be
            stored. If none given or passed None, _evaluate_ne will return
            the result of the calculation. Otherwise, nothing is returned
            from this method and the results are stored in the indicated variable.
            This has an important effect in performance for large arrays.
        """
        assert isinstance(ne_ex, str)
        if local_vars is None:
            local_vars = {}
        local_vars.update(self._inputs_to_dict())
        if out_var is None:
            return ne.evaluate(ne_ex, local_dict=local_vars)
        else:
            out = getattr(self, out_var)
            try:
                ne.evaluate(ne_ex, local_dict=local_vars, out=out)
            except:
                raise

class TimeDependentParameter(Variable):
    """This class represents a deterministic but timedependent variable.
    """
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
