#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:40:01 2017

@author: luk-f-a
"""
import xarray as xr
from .utils import if_delegate_has_method, ConsistencyError



class Pipeline(object):
    """Pipeline of transforms with an initial generator.
    
    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    """
    
    def __init__(self, steps):
        # shallow copy of steps
        self.steps = steps
        self._validate_steps()
    
    def _validate_steps(self):
        assert hasattr(self.steps[0], 'generate')
        
    def generate(self):   
        X = self.steps[0].generate()
        for step in self.steps[1:]:
            X = step.transform(X)
        return X


class ModelUnion(object):
    """Merges the outputs of two or more models
    

    Parameters
    ----------
    input_list : list of (string, model) tuples
        List of model objects where the data is coming from. The first
        half of each tuple is the name of the model.

        
        
    """
    __slots__ = ('type', '_first_model', 'model_list')
    
    def __init__(self, model_list):
        self.model_list = model_list
        self._validate_models()
        
    def _validate_models(self):
        """checks given models are valid
        
        checks:  all objects are model-like
                 svar names are not repeated
                 all inputs are either numpy or xarray based, checking svar property
        """
        gen = True
        transf = True
        for name, model in self.model_list:
            if hasattr(model, 'generate') and gen is False:
                raise ConsistencyError()
                
            if hasattr(model, 'transform') and transf is False:
                raise ConsistencyError()
                
            if hasattr(model, 'generate'):
                transf = False
                self.type = 'generator'
                
            if hasattr(model, 'transform'):
                gen = False
                self.type = 'transformer'
                
            if not (hasattr(model, 'transform') or hasattr(model, 'generate')):
                #TODO: this error is probably not the right one
                raise ValueError()
                
        self._first_model = self.model_list[0][1]
            
        #TODO: continue checks
    
    
    @if_delegate_has_method(delegate='_first_model')
    def generate(self):
        data = []
        for name, model in self.model_list:
            data.append(model.generate())
        data = xr.concat(data, dim='svar')
        return data

    @if_delegate_has_method(delegate='_first_model')
    def transform(self):
        data = []
        for name, model in self.model_list:
            data.append(model.transform())
        data = xr.concat(data, dim='svar')
        return data
        
    @property
    def svars(self):
        svars = []
        for i in self.input_list:
            svars.extend(i.svars)
        return svars
    
    
    
