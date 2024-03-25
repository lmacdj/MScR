#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:42:36 2024

@author: vsbh19
"""

import numpy as np
def get_metadata(file, station_name):
    
    try:
        stations = file.keys()
    #global attributes
        attributes = np.empty((len(stations), 4))
        for i, x in enumerate(stations):
            try:
                attributes[i, :] = [float(file[x].attrs[u]) for u in [i for i in file[x].attrs.keys()]]
            except: 
                print("Attributes cannot be satisfied")
        #global time 
        try: 
            mag,time = [i[1] for i in file.attrs.items() if i[0] in ["time", "mag"]]
        except: 
            print("Time and mag not available big rip")
            print([i for i in file.attrs.items()])
        
        try:
            attribute_names = [r for r in file[station_name].attrs.keys()]
        except: 
            print("Atrributes cannot be received")
        attribute_names.append("Epicentral time")
        print(file[x].attrs.keys())
    #attributes[:,0] = attribute_names
        return attribute_names, attributes, time, mag
    except ValueError: 
        print(file[x].attrs.keys())
        print("File Metadata not consistent"); return None, None, None, None