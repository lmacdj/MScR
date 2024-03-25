#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:32:58 2024

@author: vsbh19
INSTRUMENT RESPONSE FOR HI NET NETWORK 

from https://seisman.github.io/HinetPy/appendix/response.html
parameters from https://seisman.github.io/HinetPy/appendix/channeltable.html

"""
import numpy as np
import matplotlib.pyplot as plt


G = 175.6 #GAIN FACTOR IN MS^-1
wf = 1.0 #NATURAL  RESONANT  FREQUENCY  (ANGULAR)
h = 0.7 #DAMPENING CONSTANT 

f = np.logspace(np.log10(1e-3), np.log10(50), 100) #pPLOTTING SPACE
s = f * 2 * np.pi #CONVERT TO USUAL CO-ORDINATES

res = []
for i,ss in enumerate(s):
    res.append(G*ss**2 / ( ss**2 + 2*wf*h*ss + wf**2)) #TRASNFER FUNCTION
res = np.array(res)
res /= max(res)

fig,ax = plt.subplots()
ax.plot(f, res)
ax.set_xscale('log');ax.set_yscale('log')
ax.set_xlim([1e-3,50])
ax.set_ylim([1e-4,1.2])
ax.set_xlabel('Natural Frequency (Hz)')
ax.set_ylabel('Normalised Response')
plt.title("Instrument Response")

