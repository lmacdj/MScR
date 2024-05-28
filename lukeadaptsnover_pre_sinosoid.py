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
from scipy import signal 
import h5py
import sys
from sklearn.preprocessing import normalize
import h5py 
import math
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
def sin(timeto, number, samplerate, onofflength, intensity, amplitude, wavelets, summed_frequencies, dampening, norm):
    snippets = np.zeros((number, timeto*samplerate))
    indices = []
    x = np.linspace(0, timeto, timeto * samplerate)
    length = len(x)
    onoff_idx = np.random.randint(timeto * samplerate, size=int(onofflength))
    freq_values = []
    if wavelets==True: 
        wavelet_widths = np.random.randint(20,50, (5,))
        #IDEALLY WE WANT IT TO CLUSTER ON THE INTENSITY OF THE WAVELETS
    global frequencies_f, array, wavelet
    ###LETS CREATE EACH SEISMOGRAM
    
    global truths
    truths = np.zeros((number,)) #the truths array
    for i in range(number):
        #if wavelets == True: 
        global random_selection; random_selection = np.random.randint(0,4)
        
        array = np.zeros((length,)) # the data array
        if random_selection == 0:
            
            #if np.random.randint(0,2) == 0: 
            wavelet = signal.ricker(2000, np.random.choice(wavelet_widths)) * 0.1
            
            # plt.plot(wavelet); sys.exit()
            truths[i] = 1 
            global random_point
            random_point = np.random.randint(length - len(wavelet), size = None)
 
            
            array[random_point:random_point+len(wavelet)] =  wavelet / np.max(np.abs(wavelet)) #normalisation
            print("amp wavelet", np.max(array[random_point:random_point+len(wavelet)]))
            #plt.plot(array); sys.exit()
            #else: 
                #truths[i] = 0
            #frequency = np.random.normal(3,1)
            #sin_wavelet = np.sin(frequency*2 *x* np.pi ) #* dampening#3 Hz background freequency 
           
        #if summed_frequencies == True:
        elif random_selection == 1:
            frequency1 = random.uniform(1.5, 3)  # Adjust the range as needed
            frequency2 = random.uniform(0.2,1.5)
            
            array[:] = np.sin(x * frequency1 * 2 * np.pi)
            print("amp sinosoid", np.max(array))
            array[:]+= np.sin(x * frequency2 * 2 * np.pi)
            truths[i] = 2 if truth_ultra == True else 1 
        elif random_selection == 2:
            frequencies_f = np.random.randint(0,3, (5,)) #5 clusters of consistent frequencies
            #clusters = np.randint(0)
            modulation_frequency = 0.003
            modulation_signal = np.sin(2 * np.pi * modulation_frequency * x)
            carrier_frequency =  modulation_signal * 1 + 1 #* frequencies_f[0] * 0.5
            #array[:] = np.sin(x*frequencies_f[np.random.randint(frequencies_f.shape[0])]*2*np.pi)
            array[:] = 5*np.sin(x*carrier_frequency*2*np.pi)
            print("amp mod", np.max(array[:]))
            truths[i] = 3 if truth_ultra == True else 1 
        elif random_selection == 3:
            array[:] = np.zeros(len(x))
            truths[i] = 0 if truth_ultra == True else 1 
        ones = np.ones(length)
        zeros = np.zeros(int(onofflength*samplerate))
        # for u in onoff_idx:
        #     try:
                
        #         ones[u:u + int(onofflength * samplerate)] = zeros[:]
        #     except:
        #         break
        #         #print("g")
        global noise
        noise = np.random.normal(scale = np.max(array[:]), size = length)#mu, sigma, samples
        intense_factor = np.random.choice(intensity) if noise_stamp == True else intensity
        print("noise bassline amp", np.max(noise), random_selection)
        print("S-N R", np.max(array)/np.max(noise), random_selection)
        noise *= intense_factor
        array+= noise
        indices.append(intense_factor)
        # add = normalize([array], norm = norm) #normalise requires 2d shapoe
        # add = np.reshape (array, (length,)) #reshape it back to original
        snippets[i,:] = array
    snippets = normalize(snippets, norm = norm)
    return x, snippets, truths , indices 
####------------------------PARAMETERS FOR SINOSOID-------------------------------
#%% Generate 40 seconds snippets with random frequencies
timeto = 240
number = 100
samplerate = 100
onofflength = 200
amplitude = 4 
background_damp = 0.25

summed_frequencies = True
differing_frequencies = True
wavelets = True 
noise_stamp = True 
#%% 
if noise_stamp == True: 
    intensity = [np.around(i,decimals =2) for i in np.logspace(-2,5,num = 10, dtype = float)] #analysis of how sensitive the system is to noise 
else:
    intensity = 4 #noise intensity
seg =int(720*(timeto/40)) #for SHAPE=(None, 140, 41, 1)
lap =int(639*(timeto/40))#from the snover study of 90s overlap
if all([summed_frequencies, differing_frequencies,wavelets]): #if all the variables are true
    truth_ultra = True 
#------------------------------------------------------------------------------------
#sys.exit()
x, sins, truths, indices = sin(timeto, number, samplerate, onofflength, intensity, amplitude,wavelets, summed_frequencies, background_damp, norm = "l2")
frequencies, times, Sxx = spectrogram(sins[0], 100, nperseg = seg, noverlap = lap, window = "blackman")
fre_cro_i = 140 #was 140  # indexes the data so that it is CROPPED to 45Hz (originally), now at 24 hertz
Sxx = Sxx[:fre_cro_i]
frequencies = frequencies[:fre_cro_i]
# for i in range(0,4):
#     instance = np.where(truths == i)[0]
#     fig = plt.figure(figsize=(29,8)) 
#     fig.add_subplot(1,2,1)
#     plt.pcolormesh(times, frequencies, Sxx)
#     fig.add_subplot(5,2,1)
#     plt.title(f"Spectrogram for truths{truths[instance[0]]}")
#     plt.plot(sins[instance[0]]);# sys.exit(0)
#     plt.savefig(f"/home/vsbh19/plots/Clusters_station/Sinosoid.h5/Example_of_instance_{i}")
#]plt.close(fig)
# TASK INCEASE WINDOW LENGTH TO GET BETTER REPRESENTATION OF LOWER FREQUENCY 


fig, axes = plt.subplots(2, 1, figsize=(29,8)) 

axes[0].pcolormesh(times, frequencies, Sxx)

plt.title(f"Spectrogram for truths{truths[0]} and {indices[0]}")
axes[1].plot(sins[0]);
#print(indices)
sys.exit(0)
stations = np.ones(number)
tobe = np.zeros((number, len(frequencies), len(times)))
tobe[0, :, :] = Sxx
# ^initial run to get sizes of frequencies and time
#indices.append(1)
print("Spectrograms have been created")
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
with h5py.File(f"/nobackup/vsbh19/h5_files/%sSpectrograms_Sinosoid_Sinosoid_[2]_{timeto}_l2_training.h5" %("NOISE_STA" if noise_stamp else "") , "w") as f:
    f.create_dataset("Sinosoid", data = tobe) #uploads the data into the h5 file
    locset = f.create_dataset("Indices", data = indices)
    f.create_dataset("Stations", data = stations)
    f["Frequency scale"] = np.array(frequencies)
    f["Frequency scale"].make_scale("Frequency scale")
    f["Sinosoid"].dims[1].attach_scale(f["Frequency scale"])
    f["Time scale"] = np.array(times)
    f["Time scale"].make_scale("Time scale")
    f["Sinosoid"].dims[0].attach_scale(f["Time scale"])
    try: 
        f.create_dataset("Truths", data = truths)
    except: 
        print(f"NO truth data to --> /nobackup/vsbh19/h5_files/Spectrograms_Sinosoid_Sinosoid_[2]_{timeto}_training.h5")