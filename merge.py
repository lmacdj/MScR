#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:11:35 2024

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
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
global sta
#%%
'''
1 August 1002
'''
date = "May2005"
directory = f"/nobackup/vsbh19/HiNetDownload{date}/"
outdir = f"/nobackup/vsbh19/HiNetDownload{date}/MSeed/"
os.chdir(directory)
directories = [x[1] for x in os.walk(directory)] 
u = directories[0][0] if directories[0][0] == "SAC3" else directories[0][1]
new_dir = directory + u
print("hello")
if os.path.exists(outdir) == False: 
    dir_new = "/MSeed/"
    parent_dir = directory
    path = parent_dir + dir_new
    os.mkdir(path) 

for i in directories[2][:]: 
    os.chdir(directory + u + "/" + i) 
    filelist = glob.glob("*.sac") 
    print(i)
    dic = {}  
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
    for k in sorted_dic.keys():
        #print(k)
        fico = k+"*.sac"
        flist = glob.glob(fico)
        sta = read(flist[0]) # read first file
        for file in flist[1:]: # read file 2 to end and stack them in sta
            sta  += read(file)
        sta.sort(['starttime']) # sort by time
        t0 = sta[0].stats.starttime # get initial time
        sta.merge(method=1) # merge in single stream
        stri=k+str(sta[0].stats.starttime)+str(sta[0].stats.endtime)
        stra=outdir+stri.replace(':','-')+".mseed"
        for tr in sta: 
            if isinstance(tr.data, np.ma.masked_array):
                tr.data = tr.data.filled()
        try: 
            sta.write(stra,format= 'MSEED')
        except: 
            dir_new = "/MSeed/"
            parent_dir = directory
            path = parent_dir + dir_new
            os.mkdir(path) 
            
            sta.write(stra,format= 'MSEED')
#%% Day drumplot:
sys.exit()
sta.plot(type='dayplot', interval = 60, title = 'start='+str(t0)[0:-8])
sys.exit()
#%% 
'''
15 August 1002
'''
outdir = '/home/stefan/HiNetDownload/Data_from_2001/2001/08/15/01/01/'
indir = outdir + 'V/M/SAC'
os.chdir(indir)
filelist = glob.glob("*.SAC") 
dic = {}  
for x in filelist:  
    key = x[:8] # The key is the first 16 characters of the file name
    group = dic.get(key,[])
    group.append(x)  
    dic[key] = group
#%%
myKeys = list(dic.keys())
myKeys.sort()
sorted_dic = {i: dic[i] for i in myKeys}
#%%
for k in sorted_dic.keys():
    print(k)
    fico = k+"*.SAC"
    flist = glob.glob(fico)
    sta = read(flist[0]) # read first file
    for file in flist[1:]: # read file 2 to end and stack them in sta
        sta  += read(file)
    sta.sort(['starttime']) # sort by time
    t0 = sta[0].stats.starttime # get initial time
    sta.merge(method=1) # merge in single stream
    stri=k+str(sta[0].stats.starttime)+str(sta[0].stats.endtime)
    stra=outdir+stri.replace(':','-')+".mseed"
    sta.write(stra,format= 'MSEED')
#%% Day drumplot:
sta.plot(type='dayplot', interval = 60, title = 'start='+str(t0)[0:-8])
#%% 
'''
17 August 1002
'''
outdir = '/home/stefan/HiNetDownload/Data_from_2001/2001/08/17/01/01/'
indir = outdir + 'V/M/SAC'
os.chdir(indir)
filelist = glob.glob("*.SAC") 
dic = {}  
for x in filelist:  
    key = x[:8] # The key is the first 16 characters of the file name
    group = dic.get(key,[])
    group.append(x)  
    dic[key] = group
#%%
myKeys = list(dic.keys())
myKeys.sort()
sorted_dic = {i: dic[i] for i in myKeys}
#%%
for k in sorted_dic.keys():
    print(k)
    fico = k+"*.SAC"
    flist = glob.glob(fico)
    sta = read(flist[0]) # read first file
    for file in flist[1:]: # read file 2 to end and stack them in sta
        sta  += read(file)
    sta.sort(['starttime']) # sort by time
    t0 = sta[0].stats.starttime # get initial time
    sta.merge(method=1) # merge in single stream
    stri=k+str(sta[0].stats.starttime)+str(sta[0].stats.endtime)
    stra=outdir+stri.replace(':','-')+".mseed"
    sta.write(stra,format= 'MSEED')
#%% Day drumplot:
sta.plot(type='dayplot', interval = 60, title = 'start='+str(t0)[0:-8])
#%% 