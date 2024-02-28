#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:34:44 2023

@author: vsbh19
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import h5py
import sys
#random.seed(812)
# def sin(timeto, onoff, samplerate, onofflength, frequency):
#     x = np.linspace(0,timeto, timeto*samplerate)
#     sins = np.sin(x*frequency*2*np.pi)
#     length = len(x)
#     ones = np.ones(length)
#     onoff = np.random.randint(timeto*samplerate, size = onoff) 
#     zeros = np.zeros(int(samplerate*onofflength)) #zeros to nullify the one array
#     for u in onoff:
#         print(u)
#         try:
#             ones[u:u+onofflength*samplerate] = zeros[:]#nullifying the one array
#         except: 
#             break
#     sins *= ones #adds some permetation 
#     return x, sins


# x, sins = sin(40, 10, 100, 3,20) #timeto, onoff, samplerate, onoff length, frequency all in seconds
# #print(x.shape, sins.shape)
def sin(timeto, onoff, samplerate, onofflength, intensity):
    snippets = np.zeros((number, timeto*samplerate))
    freq_values = []
    global frequencies_f
    for i in range(onoff):
        if summed_frequencies == True:
            frequency1 = random.uniform(3, 8)  # Adjust the range as needed
            frequency2 = random.uniform(0.2,4)
            x = np.linspace(0, timeto, timeto * samplerate)
            sins = np.sin(x * frequency1 * 2 * np.pi)
            sins+= np.sin(x * frequency2 * 2 * np.pi)
            length = len(x)
        elif differing_frequencies == True: 
            frequencies_f = np.random.randint(0,15, (5,)) #5 clusters of consistent frequencies
            #clusters = np.randint(0)
            x = np.linspace(0, timeto, timeto * samplerate)
            sins = np.sin(x*frequencies_f[np.random.randint(frequencies_f.shape[0])]*2*np.pi)
            length = len(x)
        ones = np.ones(length)
            
        onoff_idx = np.random.randint(timeto * samplerate, size=int(onofflength))

        zeros = np.zeros(int(onofflength*samplerate))
        for u in onoff_idx:
            try:
                
                ones[u:u + int(onofflength * samplerate)] = zeros[:]
            except:
                break
                #print("g")

        sins *= ones
        noise = intensity*np.random.normal(0, 2.5,length) #mu, sigma, samples
        sins += noise
        snippets[i,:] = sins 
        

    return x, snippets
####------------------------PARAMETERS FOR SINOSOID-------------------------------
# Generate 40 seconds snippets with random frequencies
timeto = 40
number = 10000
samplerate = 100
onofflength = 0
summed_frequencies = False
differing_frequencies = True
intensity = 0 #noise intensity
seg = 720 #width of window segment 
lap = 639 #amount of overlap

#------------------------------------------------------------------------------------
#sys.exit()
x, sins = sin(timeto, number, samplerate, onofflength, intensity)
frequencies, times, Sxx = spectrogram(sins[0], 100, nperseg = seg, noverlap = lap, window = "blackman")
fre_cro_i = 140 #was 140  # indexes the data so that it is CROPPED to 45Hz (originally), now at 24 hertz
Sxx = Sxx[:fre_cro_i]
frequencies = frequencies[:fre_cro_i]
fig = plt.figure(figsize=(29,8)) 
fig.add_subplot(1,2,1)
plt.pcolormesh(times, frequencies, Sxx)
fig.add_subplot(5,2,1)
plt.plot(x,sins[10])#; sys.exit(0)
#]plt.close(fig)
# TASK INCEASE WINDOW LENGTH TO GET BETTER REPRESENTATION OF LOWER FREQUENCY



tobe = np.zeros((number, len(frequencies), len(times)))
tobe[0, :, :] = Sxx
# ^initial run to get sizes of frequencies and time
indices = np.linspace(1,number, num=number, dtype = int)
stations = np.linspace(1,number, num = number, dtype = int)
#indices.append(1)
print("good")
for i,u in enumerate(sins[1:]):
    # print(u)
    #print(u[1], len(u)); sys.exit()
    #indices.append(i+2) #synthetic indices
    frequencies, times, Sxx = spectrogram(sins[i], 100,
                                          nperseg=seg, noverlap=lap,
                                          window="blackman")
    Sxx = Sxx[:fre_cro_i]
    frequencies = frequencies[:fre_cro_i]
    #sys.exit()
    tobe[i, :, :] = Sxx
    # print(Sxx)
    # sys.exit()

if len(indices) != number: 
    print("get a life loser xxx"); sys.exit()
print(indices)
with h5py.File(f"/nobackup/vsbh19/h5_files/Spectrograms_Sinosoid_Sinosoid_[2]_{timeto}_training.h5" , "w") as f:
    f.create_dataset("Sinosoid", data = tobe) #uploads the data into the h5 file
    locset = f.create_dataset("Indices", data = indices)
    f.create_dataset("Stations", data = stations)
    f["Frequency scale"] = np.array(frequencies)
    f["Frequency scale"].make_scale("Frequency scale")
    f["Sinosoid"].dims[1].attach_scale(f["Frequency scale"])
    f["Time scale"] = np.array(times)
    f["Time scale"].make_scale("Time scale")
    f["Sinosoid"].dims[0].attach_scale(f["Time scale"])