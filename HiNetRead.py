#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:52:55 2024

@author: vsbh19
"""

"""
HI NET READ 

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:23:57 2024

@author: stefan
"""
from obspy import read; import matplotlib.pyplot as plt; import numpy as np
from obspy import UTCDateTime
import glob; import os; import sys; from scipy.signal import spectrogram #import h5py
import math; from sklearn.preprocessing import normalize; import h5py
#import time
from butter_filter import butter_filter; import copy
#def hinetread(**kwargs):
    #%% rea all in folder 2008-03-08 and plot single example:
try: 
    date = sys.argv[1]
    overlap = float(sys.argv[2])
    date =  date.replace(",","")
except Exception as e:
    print("RUNNING BACKUP VARIABLES BECAUSE OF", e)
    date = "May2005"
    overlap = 0.25
duration = 120
beta = np.pi*1.8
#directory = kwargs.get("dir") if kwargs.get("dir") is not None else f'/nobackup/vsbh19/HiNetDownload{date}/SAC3'
directory = f'/nobackup/vsbh19/HiNetDownload{date}/MSeed'
out_directory = f'/nobackup/vsbh19/h5_files/'
os.chdir(directory)
filelist = glob.glob("*.mseed") 
dic = {}  
basedate = UTCDateTime(2001,1,1); sys.exit()
for x in filelist:  
    key = x[:8] # The key is the first 16 characters of the file name
    group = dic.get(key,[])
    group.append(x)  
    dic[key] = group

#%%
myKeys = list(dic.keys())
myKeys.sort()
sorted_dic = {i: dic[i] for i in myKeys}
#sys.exit()
#%%
seg =390 #for SHAPE=(None, 140, 41, 1)
lap =int(seg*0.25)

component = "U"
iteration = [i for i in sorted_dic.keys() if i[7] == "U"]
station_dic = {}
Xbig = np.empty
for i,k in enumerate(iteration): #FOR EVERY STATION
    #print(k)
    fico = k[:-1]+"U*.mseed" #we only want the vertical component
    flist = glob.glob(fico)
    sta = read(flist[0]) # read first file
    
    
    for file in flist[1:]:
        sta  += read(file)
    sta.sort(['starttime'])
    
    t0 = sta[0].stats.starttime 
    t0_original = copy.deepcopy(t0) #deep copy t0 such that it doesnt get overwritten!!
    tend = sta[-1].stats.endtime #; sys.exit()
    number_spectrograms = (int(math.ceil(tend-t0)/(duration*(1-overlap))))
    tobe = np.zeros((1, 140, 40))#; sys.exit()
    poor = 0
    time_array, station_array = [],[]
    station = flist[0][2:5]
    station_dic[i] = station
    for i in np.arange(number_spectrograms): #FOR EVERY SPECTROGRAM
    
        t0 += duration*(1-overlap)
        t1 = t0 + duration
        #print(t1-t0)
        
        slo = sta.slice(t0, t1)
        
        trace_data_list  = []
        for trace in slo: 
            #print(trace.data); sys.exit()
            trace_data_list = trace.data
            #print(trace.data, "!!")
        
        #trace_data_list = np.array(trace_data_list)
        #print(trace_data_list)
        if True in trace_data_list or len(trace_data_list) < (duration*100): #if the array is a masked array
            #tobe[i,:,:] = np.full((140,41), 0, dtype =object) #null array due to it being filled
            poor += 1
            continue #the dataset is corrupted go onto the next dataset
        #trace_data_list = np.reshape(trace_data_list, (len(trace_data_list[0]),))
        add = butter_filter(trace_data_list, "high",2,100, 3, False)
        add = normalize([add], norm = "l2")
        frequencies, times, Sxx = spectrogram(add, 100,
                                          nperseg=seg, noverlap=lap,
                                          window=("kaiser",  beta))#,  nfft = seg*4)
        #sys.exit()
        fre_cro_i = 140 #was 140  # indexes the data so that it is CROPPED to 45Hz (originally), now at 24 hertz
        Sxx = np.reshape(Sxx, (len(Sxx[0,:,0]), len(Sxx[0,0,:])))
        #Sxx = Sxx[8:8+fre_cro_i]
        Sxx = Sxx[:fre_cro_i]
        #frequencies = frequencies[8:8+fre_cro_i] #GET RID OF THE 1HZ BAND
        frequencies = frequencies[:fre_cro_i]
        # if i%10 == 0:
        #     plt.pcolormesh(Sxx[0, 30:])
        #     time.sleep(2)
        if i != 1:
            #Sxx_fit = (1,) + Sxx.shape
            tobe = np.concatenate((tobe, np.reshape(Sxx, (1,) + Sxx.shape)), axis=0)
            time_array.append(t0 - t0_original)#; station_array.append(i) #uploads the index which belongs to station
        else:
            time_array.append(t0 - t0_original)#; station_array.append(i) #uploads index which belongs to station
            tobe[i-1,:,:] = Sxx
    with h5py.File(f"{out_directory}Tremor_Spectrograms_all_{component}_{duration}_l2_training.h5","a") as f:
        try:
            f.create_dataset(f"{date}+{k}", data = tobe)
            f.create_dataset(f"{date}+{k}+times", data = time_array)
            f.create_dataset(f"{date}+{k}+station", data = [station for i in time_array])
        except: 
            f.close() #for if the file already exists 
            with h5py.File(f"{out_directory}Tremor_Spectrograms_all_{component}_{duration}_l2_training.h5","w") as f:
                f.create_dataset(f"{date}+{k}", data = tobe)
                f.create_dataset(f"{date}+{k}+times", data = time_array)
                f.create_dataset(f"{date}+{k}+station", data = [station for i in time_array])
                
                
        #sys.exit()
    print(f"poor rate for {k}=", 100* poor /number_spectrograms)
    #time.sleep(3) 
    #sys.exit()
    

#print(directories)
#st = read('./', debug_headers=True)
#print(st)
# st[0].plot()
# st[2].plot()
# st[9].plot(type='dayplot', interval = 4)
#%% plot all traces in st;
# for tr in st:
#     tr.plot()
sys.exit()
#%% 
''' stack all streams of 2008 March 8 together;
    order them by time; merge; drumplot: '''
flist = glob.glob('./SAC3/2008-03-08/N.IKTH.N*')
sta = read(flist[0])


sta.merge(method=1)
sta.plot(type='dayplot', interval = 60, title = 'start='+str(t0))
#%%
''' PLOT ONE MINUTE DATA CLOSE TO START AND END OF DAY
    --------------------------------------------------
    the tremor becomes much weaker toward the end of the day, 
    mayve as a consequence of earthquake during 4th hour. 
    I select two slices, one with strong tremor and one without:
    one minute extract from day 1 and from day 22'''
st1 = sta.slice(t0, t0+60)
st2 = sta.slice(t0+21*3600, t0+21*3600+60)
st1.plot()
st2.plot()
dt = st1[0].stats.delta
NFFT = 256  # the length of the windowing segments
noverlap= 255
Fs = 1/dt  # the sampling frequency
scale = 'linear';#'linear' # 'dB'
#%%
'''
Spectrograms with pyplot. These allow to common scale even with dB scale'''
spl = st1[0]; 
NN = len(spl)
t = np.arange(spl.stats.npts) / spl.stats.sampling_rate
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(t, spl.data, 'k-');#ax1.set_ylim(-140,140)
ax1.set_ylabel('')
ax1.set_title(str(t0)[0:-8])
Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                    scale = scale, noverlap=noverlap,Fs=Fs)
if scale == 'linear': ma = np.max(Pxx); mi = np.min(Pxx)
else: mi = 10*np.log10(np.min(Pxx)); ma=10*np.log10(np.max(Pxx)) 
Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                    scale = scale, noverlap=noverlap, Fs=Fs, 
                                    vmin= mi, vmax = ma)
ax2.set_yscale('log')
ax2.set_ylim(.5,50);ax2.set_xlim(t[0],t[-1])
ax2.set_ylabel('freq (Hz)');ax2.set_xlabel('time (s)')
#%%
spl = st2[0]
NN = len(spl)
t = np.arange(spl.stats.npts) / spl.stats.sampling_rate
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(t, spl.data, 'k-');#ax1.set_ylim(-140,140)
ax1.set_ylabel('')
Pxx, freqs, bins, im = ax2.specgram(spl.data, NFFT=NFFT, 
                                    scale = 'linear',noverlap = noverlap, Fs=Fs,
                                    vmin= mi, vmax = ma)
ax2.set_yscale('log')
ax2.set_ylim(.5,50);ax2.set_xlim(t[0],t[-1])
ax2.set_ylabel('freq (Hz)');ax2.set_xlabel('time (s)')
''' 
In conclusion, clearly stronger tremor band <2Hz for st2'''
#%%


#hinetread()

