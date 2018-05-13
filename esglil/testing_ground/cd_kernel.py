#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:00:42 2018

@author: lucio
"""
import numpy as np

from numpy.polynomial import hermite_e as her

print(her.hermeval2d(2,3,np.array([[1,1],[1,1]])))

