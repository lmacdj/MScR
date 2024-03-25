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
    x = np.linspace(0, timeto, timeto * samplerate)
    length = len(x)
    onoff_idx = np.random.randint(timeto * samplerate, size=int(onofflength))
    freq_values = []
    if wavelets==True: 
        wavelet_widths = np.random.randint(1,50, (5,))
        #IDEALLY WE WANT IT TO CLUSTER ON THE INTENSITY OF THE WAVELETS
    global frequencies_f, array, wavelet
    ###LETS CREATE EACH SEISMOGRAM
    global truths; truths = np.ones((number,))
    
    for i in range(number):
        #if wavelets == True: 
        global random_selection; random_selection = np.random.randint(0,3)
        if random_selection == 0:
            array = np.zeros((length,))
            if np.random.randint(0,2) == 0: 
                wavelet = signal.ricker(2000, np.random.choice(wavelet_widths)) * 0.1
            # plt.plot(wavelet); sys.exit()
                truths[i] = 1 
                global random_point
                random_point = np.random.randint(length - len(wavelet), size = None)
 
            
                array[random_point:random_point+len(wavelet)] = wavelet / np.max(np.abs(wavelet)) #normalisation
            else: 
                truths[i] = 0
            #frequency = np.random.normal(3,1)
            #sin_wavelet = np.sin(frequency*2 *x* np.pi ) #* dampening#3 Hz background freequency 
           
        #if summed_frequencies == True:
        elif random_selection == 1:
            frequency1 = random.uniform(3, 8)  # Adjust the range as needed
            frequency2 = random.uniform(0.2,4)
            
            sins = np.sin(x * frequency1 * 2 * np.pi)
            sins+= np.sin(x * frequency2 * 2 * np.pi)
           
        elif random_selection == 2:
            frequencies_f = np.random.randint(0,15, (5,)) #5 clusters of consistent frequencies
            #clusters = np.randint(0)
            
            sins = np.sin(x*frequencies_f[np.random.randint(frequencies_f.shape[0])]*2*np.pi)
        elif random_selection == 3:
            sins = np.zeros(len(x))
        ones = np.ones(length)
        zeros = np.zeros(int(onofflength*samplerate))
        # for u in onoff_idx:
        #     try:
                
        #         ones[u:u + int(onofflength * samplerate)] = zeros[:]
        #     except:
        #         break
        #         #print("g")
        global noise
        noise = intensity*np.random.normal(0, 9*(length/40),length) #mu, sigma, samples
        try:
            sins *= ones * dampening
            
            sins += noise
           
        except: 
            pass
            
        
        try: 
            array+= noise
            
            #array+= sin_wavelet
            
            snippets[i,:] = array
        except Exception as e: 
            print(e)
            snippets[i,:] = sins 
        #snippets /= np.max(np.abs(snippets))
        
    snippets = normalize(snippets, norm = norm )
    return x, snippets 
####------------------------PARAMETERS FOR SINOSOID-------------------------------
# Generate 40 seconds snippets with random frequencies
timeto = 240
number = 40000
samplerate = 100
onofflength = 200
amplitude = 4 
background_damp = 0.25

summed_frequencies = True
differing_frequencies = True
#global wavelet
wavelets = True 
intensity = 1E-9 #noise intensity
seg =int(720*(timeto/40)) #for SHAPE=(None, 140, 41, 1)
lap =int(639*(timeto/40))#from the snover study of 90s overlap

#------------------------------------------------------------------------------------
#sys.exit()
x, sins = sin(timeto, number, samplerate, onofflength, intensity, amplitude,wavelets, summed_frequencies, background_damp, norm = "l2")
frequencies, times, Sxx = spectrogram(sins[0], 100, nperseg = seg, noverlap = lap, window = "blackman")
fre_cro_i = 140 #was 140  # indexes the data so that it is CROPPED to 45Hz (originally), now at 24 hertz
Sxx = Sxx[:fre_cro_i]
frequencies = frequencies[:fre_cro_i]
fig = plt.figure(figsize=(29,8)) 
fig.add_subplot(1,2,1)
plt.pcolormesh(times, frequencies, Sxx)
fig.add_subplot(5,2,1)
plt.title(f"Spectrogram for truths{truths[0]}")
plt.plot(sins[0])#; sys.exit(0)
#]plt.close(fig)
# TASK INCEASE WINDOW LENGTH TO GET BETTER REPRESENTATION OF LOWER FREQUENCY 



tobe = np.zeros((number, len(frequencies), len(times)))
tobe[0, :, :] = Sxx
# ^initial run to get sizes of frequencies and time
indices = np.linspace(1,number, num=number, dtype = int)
stations = np.linspace(1,number, num = number, dtype = int)
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
with h5py.File(f"/nobackup/vsbh19/h5_files/Spectrograms_Sinosoid_Sinosoid_[2]_{timeto}_max_training.h5" , "w") as f:
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