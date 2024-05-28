#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:23:57 2024

@author: stefan
"""
from obspy import UTCDateTime
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
#%%
#############################

def tremoryes():
    '''output message to terminal to tell that the button is working'''
    print("YES")
    root.quit()
def tremorno():
    '''output message to terminal to tell that the button is working'''
    print("NO")
    root.quit() 
def tremorbeh():
    '''output message to terminal to tell that the button is working'''
    print("NOT SURE")
    root.quit()
def bots(root):    
    b0 = Label(root, text="TREMOR")
    b0.pack()
    b1 = Button(root, text="+", command=tremoryes)
    b1.pack()
    b2 = Button(root, text="?", command=tremorbeh)
    b2.pack()  
    b3 = Button(root, text="-", command=tremorno)
    b3.pack()   
#############################
#%%
directory = "/nobackup/vsbh19/HiNetDownload/MSeed"
os.chdir(directory)
# directories = [x[0] for x in os.walk(directory)] 
# for i in directories: 
# 	os.chdir(directory + i) 
# 	
flist = glob.glob('*.mseed')
#%%
dic = {}  
for x in flist:  
    key = x[2:5] # The key is the first 16 characters of the file name
    group = dic.get(key,[])
    group.append(x)  
    dic[key] = group
#%%
myKeys = list(dic.keys())
myKeys.sort()
sorted_dic = {i: dic[i] for i in myKeys}
#%% create window for BUTTONS:
root = Tk()  # create parent window
root.attributes("-topmost", True)
root.geometry('250x150+0+0')
bots(root)

#%%
for k in sorted_dic.keys():
    for l in dic[k]:
        if l[7] == 'E':
            print(l)
            sta = read(l)
            t0 = sta[0].stats.starttime
            station = sta[0].stats.station
            channel = sta[0].stats.channel
            title = station + "|"  + channel+ "|" + str(t0)[:-8]
            print(station)
            fig,ax = plt.subplots(num = 1, clear=True)
            frame1=plt.gca()
            sta.plot(type='dayplot', interval = 60, title = title, fig=fig)
            ax.set_yticks([]);ax.set_xticks([])            
            #fig.autofmt_xdate()
            fig.canvas.draw()
            fig.canvas.flush_events()
            #fig.canvas.manager.window.attributes('-topmost', 1)
            #%%
            ''' PLOT DIFFERENT SLICES OF DATA
                -----------------------------'''
            for tplus  in range(36):
                t2 = t0+tplus*1200; 
                st1 = sta.slice(t2, t2+1200)
                st1.detrend('demean')
                dt = st1[0].stats.delta
                NFFT = 500  # the length of the windowing segments
                noverlap= 400
                Fs = 1/dt  # the sampling frequency
                scale = 'linear';#'linear' # 'dB'
                frei = 10
                print(tplus)
                #%%
                '''
                Spectrograms with pyplot. These allow to common scale even with dB scale'''
                spl = st1[0]; 
                t = np.arange(spl.stats.npts) / spl.stats.sampling_rate
                NN = len(spl)
                fig, (ax1, ax2) = plt.subplots(num = 2, clear=True, nrows=2, sharex=True)
                ax1.plot(t, spl.data, 'k-');ax1.set_ylim(-500,500)
                ax1.set_ylabel('')
                title = station + "|" +  channel+ "|" + str(t2)[:-8]
                ax1.set_title(title)
                if scale == 'linear': mi = 1e-1; ma = 500
                else: mi = 10*np.log10(1e-1); ma=10*np.log10(300) 
                Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                                    scale = scale, noverlap=noverlap,Fs=Fs,
                                                    vmin = mi, vmax = ma)
                # if scale == 'linear': ma = np.max(Pxx[frei:]); mi = np.min(Pxx[frei:])
                # else: mi = 10*np.log10(np.min(Pxx[frei:])); ma=10*np.log10(np.max(Pxx[frei:])) 
                # Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                #                                     scale = scale, noverlap=noverlap, Fs=Fs, 
                #                                     vmin= mi, vmax = ma)
                ax2.set_yscale('log')
                ax2.set_ylim(freqs[frei],freqs[-1]);
                ax2.set_xlim(t[0],t[-1])
                ax2.set_ylabel('freq (Hz)');ax2.set_xlabel('time (s)')
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()
                #fig.canvas.manager.window.attributes('-topmost', 1)
                root.mainloop()
                #%%
            # time.sleep(3)
