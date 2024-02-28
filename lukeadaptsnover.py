#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:28:26 2023

@author: vsbh19
"""
########################################IMPORTING RELEVANT  MODULES############################################
import sys
directory = "/home/vsbh19/initial_python_files/"
#sys.path.append(directory  + "lukeadaptsnover_pre.py") #mounting the previous file

import h5py 
import numpy as np
import matplotlib.pyplot as plt

#from lukeadaptsnover_pre import frequencies, times,  number_samples_window

#frequencies = frequencies; times = times; number_samples_window = number_samples_window
train_dataname = "/nobackup/vsbh19/h5_files/Spectrograms_ev0000364000._training.h5"

with h5py.File(train_dataname, "r") as f:
    X_train = []
    for i in f.keys():
        data = f.get(str(i))
        X_train.append(np.array(data))
    times = f.get("times")
    times = np.array(times)
    times = np.reshape(times, len(times))
    frequencies = f.get("frequencies")
    frequencies = np.array(frequencies)
    frequencies = np.reshape(frequencies, len(frequencies))
    number_samples_window = f.get("number_samples_window")
    #sort this out another time
#########VISUALISING SOME SPECTROGRAMS#####################################
Sxx = X_train[20000]

# print(Sxx)
plt.pcolormesh(np.array(times), np.array(frequencies), 10*np.log10(Sxx)) #
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(label = "Intensity [dB]")
plt.show()
more 

