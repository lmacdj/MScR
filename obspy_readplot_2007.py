#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:56:20 2024

@author: vsbh19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:23:57 2024

@author: stefan
Merges hour files together into 1 day miniseed  for 
    each station 
    each component
creates spectrograms
merge hour long datasets together
"""
from obspy import UTCDateTime
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import glob
import os; import sys
#%% 

''' 
MARCH 2007 - This in principle HAS tremors
'''
directory = "/nobackup/vsbh19/HiNetDownload/SAC3"
outdir = "/nobackup/vsbh19/HiNetDownload/MSeed/"
os.chdir(directory)
flist = []
dir_ls = [x[1] for x in os.walk(directory)][0];# sys.exit()
for i in dir_ls: #FOR EVERY *DAY*
    new_dir = directory + "/"+i
    os.chdir(new_dir)
    filelist = glob.glob("*.sac")
    #filelis
    print(filelist)
    dic = {}
    for x in filelist:  
        key = x[:8] # The key is the first 16 characters of the file name
        group = dic.get(key,[])
        group.append(x)  
        dic[key] = group
    #sys.exit()
#%%
    myKeys = list(dic.keys())
    myKeys.sort()
    sorted_dic = {i: dic[i] for i in myKeys}
    #sys.exit()
    for k in sorted_dic.keys(): #FOR EVERY **STATION**
        fico = k+"*.sac"
        flist = glob.glob(fico)
        sta = read(flist[0])
        for file in flist[1:]:
            try:
                sta  += read(file)
            except Exception as e: 
                print("File", file, "cannot be read due to error", e)
        sta.sort(['starttime'])
        t0 = sta[0].stats.starttime
        dt = sta[0].stats.delta
        Fs = 1/dt  # the sampling frequency
        sta.merge(method=1, fill_value = "interpolate")
        sta.plot(type='dayplot', interval = 60, title = 'start='+str(t0))
        st1 = sta.slice(t0, t0+600)
        stri=k+str(t0)+str(sta[0].stats.endtime)
        stra = outdir+stri.replace(':','-')+".mseed"
        # for tr in sta:
        #     if isinstance(tr.data, np.ma.masked_array):
        #         tr.data = tr.data.filled()

        sta.write(stra, format = "mseed")
        st1.plot()
    
sys.exit("DATA PLOTTED")
#%% Parameters for the spectrogram
NFFT = 256  # the length of the windowing segments
noverlap= 255
scale = 'linear';#'linear' # 'dB'
frei = 3
#%%
''' PLOT TIME WINDOWS'''
ti=t0
st1 = sta.slice(ti, ti+600)
spl = st1.detrend('demean'); NN = len(spl); 
t = np.arange(spl.stats.npts) / spl.stats.sampling_rate
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
mit=5*np.max([np.max(spl.data),abs(np.min(spl.data))])
ax1.plot(t, spl.data, 'k-');#ax1.set_ylim(-240,240)
ax1.set_ylabel('');ax1.set_ylim(-mit,mit)
ax1.set_title(str(t0)[0:-8])
Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                    scale = scale, noverlap=noverlap,Fs=Fs)
if scale == 'linear': ma = np.max(Pxx[frei:]); mi = np.min(Pxx[frei:])
else: mi = 10*np.log10(np.min(Pxx[frei:])); ma=10*np.log10(np.max(Pxx[frei:]))
Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                    scale = scale, noverlap=noverlap, Fs=Fs, 
                                    vmin= mi, vmax = ma)
ax2.set_yscale('log')
ax2.set_ylim(freqs[frei],freqs[-1]);
ax2.set_xlim(t[0],t[-1])
ax2.set_ylabel('freq (Hz)');ax2.set_xlabel('time (s)')
#%%
for ni in range(23):
    ti=t0+(ni+1)*3600
    st1 = sta.slice(ti, ti+600)
    spl = st1[0].detrend('demean'); NN = len(spl); 
    t = np.arange(spl.stats.npts) / spl.stats.sampling_rate
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(t, spl.data, 'k-');#ax1.set_ylim(-240,240)
    ax1.set_ylabel('');ax1.set_ylim(-mit,mit)
    ax1.set_title(str(t0)[0:-8])
    Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                        scale = scale, noverlap=noverlap, Fs=Fs, 
                                        vmin= mi, vmax = ma)
    ax2.set_yscale('log')
    ax2.set_ylim(freqs[frei],freqs[-1]);
    ax2.set_xlim(t[0],t[-1])
    ax2.set_ylabel('freq (Hz)');ax2.set_xlabel('time (s)')
#%%
''' JANUARY 2007 -- in principle has NO  tremor'''

flist = glob.glob('./SAC3/2007-01-31/N.YNDH.E*')
sta = read(flist[0])
sta.plot(type='dayplot', interval = 60, title = 'start='+str(t0))

for file in flist[1:]:
    sta  += read(file)
sta.sort(['starttime'])
t0 = sta[0].stats.starttime
dt = sta[0].stats.delta
Fs = 1/dt  # the sampling frequency
sta.merge(method=1)
sta.plot(type='dayplot', interval = 60, title = 'start='+str(t0))
sta.write()
st1 = sta.slice(t0, t0+600)
st1.plot()
#%% Parameters for the spectrogram
NFFT = 256  # the length of the windowing segments
noverlap= 255
scale = 'linear';#'linear' # 'dB'
frei = 3
#%%
''' PLOT TIME WINDOWS'''
ti=t0
st1 = sta.slice(ti, ti+600)
spl = st1[0].detrend('demean'); NN = len(spl); 
t = np.arange(spl.stats.npts) / spl.stats.sampling_rate
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
mit=5*np.max([np.max(spl.data),abs(np.min(spl.data))])
ax1.plot(t, spl.data, 'k-');#ax1.set_ylim(-240,240)
ax1.set_ylabel('');ax1.set_ylim(-mit,mit)
ax1.set_title(str(t0)[0:-8])
Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                    scale = scale, noverlap=noverlap,Fs=Fs)
if scale == 'linear': ma = np.max(Pxx[frei:]); mi = np.min(Pxx[frei:])
else: mi = 10*np.log10(np.min(Pxx[frei:])); ma=10*np.log10(np.max(Pxx[frei:]))
Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                    scale = scale, noverlap=noverlap, Fs=Fs, 
                                    vmin= mi, vmax = ma)
ax2.set_yscale('log')
ax2.set_ylim(freqs[frei],freqs[-1]);
ax2.set_xlim(t[0],t[-1])
ax2.set_ylabel('freq (Hz)');ax2.set_xlabel('time (s)')
#%%
for ni in range(12):
    ti=t0+(ni+10)*3600
    print(ti)
    st1 = sta.slice(ti, ti+600)
    spl = st1[0].detrend('demean'); NN = len(spl); 
    t = np.arange(spl.stats.npts) / spl.stats.sampling_rate
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(t, spl.data, 'k-');#ax1.set_ylim(-240,240)
    ax1.set_ylabel('');ax1.set_ylim(-mit,mit)
    ax1.set_title(str(t0)[0:-8])
    Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                        scale = scale, noverlap=noverlap, Fs=Fs, 
                                        vmin= mi, vmax = ma)
    ax2.set_yscale('log')
    ax2.set_ylim(freqs[frei],freqs[-1]);
    ax2.set_xlim(t[0],t[-1])
    ax2.set_ylabel('freq (Hz)');ax2.set_xlabel('time (s)')
#%%
