#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:30:17 2024

@author: vsbh19
"""

import numpy as np; import matplotlib.pyplot as plt


x = np.linspace(0,10,50)
y = np.sin(x)
yy = np.cos(x)
yyy = np.sin(x-np.pi)
fig,axes = plt.subplots(nrows=3, ncols =2)
axes[0][0].plot(x,y,x,yy)
axes[1][0].plot(x,y,x,y)
axes[0][1].plot(y,yy)
axes[1][1].plot(y,y)
axes[2][0].plot(x,yyy,x,y)
axes[2][1].plot(y,yyy)