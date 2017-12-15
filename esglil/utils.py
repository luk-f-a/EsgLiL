#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:42:45 2017

@author: luk-f-a
"""




class ConsistencyError(Exception):
    """Exception raised for inconsistent inputs.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
