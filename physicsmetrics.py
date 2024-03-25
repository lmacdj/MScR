#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:43:40 2024

@author: vsbh19
"""

"""
PHYSICAL METRICS 

Contains: 
    1. Timeseries energy 
    2. Maxmimum amplitude
    3. Root mean squared amplitude 
    """
import numpy as np
class physics_metrics:
    def __init__(self, timeseries, **kwargs):
         
        self.timeseries = np.array(timeseries) * 1E-3 #assuming u is in mic-metres
        self.timeseries = np.reshape(self.timeseries, (len(timeseries),))
        self.unit = kwargs.get("unit", "mega")
    def timeseries_energy(self):
        
        energy = np.sum(self.timeseries**2)
        if self.unit == "mega": 
            return round(energy /  1E6, 2) #mega joules
        if self.unit == "kilo":
            return round(energy / 1E3, 2)
    def max_amp(self):
        return np.max(self.timeseries)
    def rms(self):
        rms = np.sqrt(np.mean(self.timeseries**2))
        return round(rms , 2)
    
    
# data = [1,2,3,4,5]

# hello = physics_metrics(data)
# print("Energy", hello.timeseries_energy())