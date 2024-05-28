# -*- coding: utf-8 -*-
"""
Tests of direct doenload from IRIS and from NIED
"""
import os; import sys
os.chdir ('/nobackup/vsbh19/HiNetDownloadMay2005')
from HinetPy import Client, win32
client = Client("stefanazzz", "5Ltk79finxrU")
client.doctor()
client.info("0101")
#%%
# client.get_event_waveform(
#     "201001010000",
#     "201001020000",
#     minmagnitude=4.0,
#     maxmagnitude=7.0,
#     mindepth=0,
#     maxdepth=70)
# data, ctable = client.get_continuous_waveform("0101", "2020301010000", 2)
# win32.extract_sac(data, ctable, suffix="", outdir="SAC")


# client.select_stations("0101", ["N.IKTH", "N.YNDH"])
# data, ctable = client.get_continuous_waveform("0101", "2004-08-17T00:00", 2)

# client.select_stations("0101", ["N.HKTH", "N.SGOH", "N.KSAH","N.YSDH"])
# data, ctable = client.get_continuous_waveform("0101", "2003-04-22T00:00", 2)

''' two stations from Shikoku Tremor 1 :'''
"""
Day  = 99.54
Lasting = 16.54 days """
#%%
#FOR TREMOR ONE (APRIL)

# client.select_stations("0101", ["N.MURH",
# "N.KTGH",
# "N.MTYH",
# "N.GHKH",
# "N.INOH" ]) 
#FOR TREMOR TWO (MAY)
client.select_stations("0101", ["N.KWBH",
"N.OOZH", 
"N.TBRH", 
"N.SJOH", 
"N.IKTH" ]) 
''' note: if > 1 hour, NIED split automatically in one-hour slices'''
#%% make 24 hours list:
from datetime import datetime, date
import numpy as np
hours = np.linspace(0,23,24,dtype=int)
year = 2005
month = 5
day_start = 6

try:
    day = int(sys.argv[1])  -1 + day_start 
except: 
    day = day_start
#%%
'''https://doi.org/10.1016/j.jog.2011.04.002 Fig 2 for the 2008 dates'''
for hour in hours:
    # a week in march in :
    starttime = datetime(year, month, day, hour)  # JST time
    tim = "."+starttime.isoformat().replace(":","-")+".sac"
    data, ctable = client.get_continuous_waveform("0101", starttime, 60)
    # save in sac format:
    try:
        win32.extract_sac(data, ctable, suffix=tim, outdir=f"SAC3/{year}-{month}-{day}")
    except: 
        directory = f"/{year}-{month}-{day}"
        parent_dir = "/nobackup/vsbh19/HiNetDownloadMay2005/SAC3"
        path = os.path.join(parent_dir, directory) 
        os.mkdir(path) 
        win32.extract_sac(data, ctable, suffix=tim, outdir=f"SAC3/{year}-{month}-{day}")
        
sys.exit(f"Completed first tremor epsiode for Day {day}")
#%%
''' two stations from Shikoku Tremor 2 :'''
"""
Day  =126.09
Lasting = 13days """
year = 2005
month = 5 
day = 6
client.select_stations("0101", ["N.KWBH",
"N.OOZH", 
"N.TBRH", 
"N.SJOH", 
"N.IKTH" ])
''' note: if > 1 hour, NIED split automatically in one-hour slices'''
# this should be "tremor-free" day, end of Jan 2008:
# for hour in hours:
#     starttime = datetime(2008, 1, 31, hour)  # JST time
#     tim = "."+starttime.isoformat().replace(":","-")+".sac"
#     data, ctable = client.get_continuous_waveform("0101", starttime, 60)
#     # save in sac format:
#     win32.extract_sac(data, ctable, suffix=tim, outdir="SAC3/2008-01-31")
# I am only extracting tremor data so this is commented out. 
# #%%
# '''
# from https://doi.org/10.1029/2008JB006048 figure 8
# this should be "silent" day: Jan 2007 January 31:'''
# for hour in hours:
#     starttime = datetime(2007, 1, 31, hour)  # JST time
#     tim = "."+starttime.isoformat().replace(":","-")+".sac"
#     data, ctable = client.get_continuous_waveform("0101", starttime, 60)
#     # save in sac format:
#     win32.extract_sac(data, ctable, suffix=tim, outdir="SAC3/2007-01-31")
# #%%
# '''
# from https://doi.org/10.1029/2008JB006048 figure 8
# this should be "tremor" day: Jan 2007 March 15:'''
# for hour in hours:
#     starttime = datetime(2007, 3, 15, hour)  # JST time
#     tim = "."+starttime.isoformat().replace(":","-")+".sac"
#     data, ctable = client.get_continuous_waveform("0101", starttime, 60)
#     # save in sac format:
#     win32.extract_sac(data, ctable, suffix=tim, outdir="SAC3/2007-03-15")
#%%
######################
######################

