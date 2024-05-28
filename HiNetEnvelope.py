#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:30:57 2024

@author: vsbh19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:23:57 2024
@author: stefan
"""
from obspy import UTCDateTime
from obspy import read
from obspy import *
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
import copy
import sys
from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from scipy.signal import find_peaks
import pandas as pd
#%%

try: 
    date = sys.argv[1]
    date=  date.replace(",","")
except:
    date = "May2005"
directory = f'/nobackup/vsbh19/HiNetDownload{date}/MSeed'
os.chdir (directory)
flist = glob.glob('*.mseed') # make list of all mseed files in folder
#%%
''' CREATE DICTINOARY W FILENAMES GROUPED BY STATION NAME AS KEY'''
dic = {}  
for x in flist:  
    key = x[:8] # the key is 3 characters of file name indicating station name
    group = dic.get(key,[])
    group.append(x)  
    dic[key] = group 
myKeys = list(dic.keys())
myKeys.sort()                                       # sort alphabetically keys
sorted_dic = {i: dic[i] for i in myKeys}            # alphabetically sorted dic
remove_peaks = True; bandpass = False
#%%
''' ALL STATIONS: DAYPLOT FILT VERT COMP, DAYPLOT ROLLING RMS'''
ifi = 0
stream = Stream()
for k in sorted_dic.keys():
    if k != 'none':                                 # what stations are skipped, invert test for stations kept
        for l in dic[k]:
            if l[7] == 'U':                         # select component
                print(l)
                sta = read(l)                       # read mseed file
                t0 = sta[0].stats.starttime         # trace from stream
                station = sta[0].stats.station
                channel = sta[0].stats.channel
                freq = int(sta[0].stats.sampling_rate)
                title = station + "|"  + channel+ "|" + str(t0)[:-8]
                ''' DETREND AND BANDPASS'''
                tra = sta[0].detrend('demean')
                #tra.filter("highpass", freq = 2)
                #tra.filter("lowpass", freq = 16)
                tra.filter("bandpass", freqmin = 2, freqmax = 16, zerophase=True)
                tra.stats.station = k
                ishi = ifi*300; ifi=ifi+1                          # figure number increment
                #fig,ax = plt.subplots(num = ifi, clear=True)
                #tra.plot(type='dayplot', interval = 60, title = title + ' FILT', fig=fig)#, color ='grey')
                #ax.set_yticks([]);ax.set_xticks([])      
                #root = fig.canvas.get_tk_widget().master
                #root.geometry('300x300+'+str(ishi)+'+0')
                #fig.canvas.draw();fig.autofmt_xdate();fig.canvas.flush_events() # update figure frame
                #
                ''' COMPUTE RMS IN ROLLING WINDOW: '''
                dff = pd.DataFrame(data = tra.data) 
                windo = 240; # seconds of rolling window
                rms_tra = np.array(dff.rolling(freq*windo,center=True).std(ddof=0))
                rms_traf = rms_tra[:,0]             # transpose
                rms_tran = np.nan_to_num(rms_traf, nan=0)  # fill gaps
                ''' CREATE TRACE FROM RMS TO USE IN OBSPY DAYPLOT '''
                tr2 = Trace(rms_tran); tr2.stats.delta = tra.stats.delta;
                tr2.stats.startttime = tra.stats.starttime
                ''' SUBSAMPLE / DECIMATE rms TRACE TO 120 s WINDOWS - no filter as already smoothed by rolling w'''
                subsamp = int(windo/tra.stats.delta); startsamp = int(subsamp/2) #subsamp number of cells per window
                rms_sub = rms_tran[startsamp::subsamp] #selects every second point
                rms_sub_old = copy.deepcopy(rms_sub)
                '''REMOVE PEAKS FROM RMS TRACE'''
                #fig,ax = plt.subplots()
                #plt.plot(rms_sub_old)
                
                if remove_peaks:
                    peaks, *_ = find_peaks(rms_sub, height =(-np.max(rms_sub), np.max(rms_sub)))
                    for i in peaks:
                        rms_sub[i] = rms_sub[i-1]
                    rms_remove_peaks = rms_sub
                    #plt.plot(rms_remove_peaks)
                    tr3 = Trace(rms_sub)
                elif bandpass: 
                    
                    tr3 = Trace(rms_sub)
                                
                    rms_bandpass = rms_sub
                    tr3.stats.sampling_rate = 100
                    tr3.filter("lowpass",freq=6, zerophase = True)
                    rms_bandpass = tr3.data
                    #plt.plot(rms_bandpass)
                #elif rolling_median:
                 
                #plt.title(f"Rempve [peaks{remove_peaks}bandpass {bandpass}")
                tr3.stats.delta = subsamp*tra.stats.delta ;
                tr3.stats.station = tra.stats.station
                tr3.stats.starttime = tra.stats.starttime + startsamp * tra.stats.delta;
                #print((tra.stats.starttime - tr3.stats.starttime) / (3600*24*100))
                stream.append(tr3)
                continue
                #sys.exit()
                
#
#                 ishi = ifi*300; ifi=ifi+1                          # figure number increment
#                 fig,ax = plt.subplots(num = ifi, clear=True)
#                 tr2.plot(num = ifi, clear=True, fig = fig)
#                 ax.set_yticks([]);ax.set_xticks([])
#                 # root = fig.canvas.get_tk_widget().master
#                 # root.geometry('300x300+'+str(ishi)+'+0')
#                 fig.canvas.draw();fig.autofmt_xdate();fig.canvas.flush_events() # update figure frame
# #
#                 ishi = ifi*300; ifi=ifi+1                          # figure number increment
#                 fig,ax = plt.subplots(num = ifi, clear=True)
#                 tr3.plot(num = ifi, clear=True, fig = fig)
#                 ax.set_yticks([]);ax.set_xticks([])
#                 # root = fig.canvas.get_tk_widget().master
#                 # root.geometry('300x300+'+str(ishi)+'+0')
#                 fig.canvas.draw();fig.autofmt_xdate();fig.canvas.flush_events() # update figure frame
#%%
                
                
                

''' SAVE RMS DECIMATED TO NEW MSEED FILES'''
stream.merge(method = 1, fill_value="interpolate")
#stream.stats.starttime = tr3.starttime
#stream.plot()
NEWDIR = ("RMS")
CHECK_FOLDER = os.path.isdir(NEWDIR)
if not CHECK_FOLDER: os.makedirs(NEWDIR)    
outfi = './'+NEWDIR+'/RMS-decim.mseed'
# st3=Stream(tr3)
stream.write(outfi, format='MSEED')
print("EXPORTED:", outfi)
#%%n
# ifi=ifi+1       
# its2 = np.linspace(0,len(tr2.data), len(tr2.data)) * tr2.stats.delta
# its3 = np.linspace(0,len(tr3.data), len(tr3.data)) * tr3.stats.delta                  
# fig,ax = plt.subplots(num = ifi, clear=True)
# ax.plot(its2, tr2.data, '-')
# ax.plot(its3, tr3.data, '-')
# ax.set_yticks([]);ax.set_xticks([])       

#%%


