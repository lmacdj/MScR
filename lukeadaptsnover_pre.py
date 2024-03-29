#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:34:52 2023

@author: vsbh19
"""

##########################################PREPROCESSING OF DATA ###############################################
###########################################IMPORT MODULES#########################################################################
import sys
import h5py
import numpy as np
#import os
#from obspy.imaging.spectrogram import spectrogram
from scipy.signal import spectrogram

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize 
import random
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
import time
import pandas as pd
import os
#import obspy
t1 = time.time()
################################################SET VAR ###########################################################################
directory = os.path.join("/nobackup/vsbh19/h5_files/")
files = ["ev0000364000.h5", "ev0000593283.h5",
         "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5", "ev0000447288.h5", "ev0000734973.h5"]
try: #PIPELINE IN SHELL SCRIPT
    filenum = int(sys.argv[1]) # Which file we want to read.
    component = [int(sys.argv[3])] #0,1,2 = X,Y,Z
    #print = type(component[0])
    print("COM", component)
    station = sys.argv[2] #station we want to read
    start_day = float(sys.argv[4])
    end_day = float(sys.argv[5])
    duration = int(sys.argv[6])
    overlap =int(sys.argv[7])
    norm=sys.argv[8]
    out_to_wav = sys.argv[9]
    #print(out_to_wav)
    print("SUCCESSFUL variable declaration of file", files[filenum])
except ValueError: 
    print("Invalid declaration of variables \n REMEMBER \nFileNum, duration and overlap are INT\nstation and norm are STRING\n start and end are FLOAT\nOut to wav BOOL")
except  IndexError: 
    print("Name Error RUNNING BACKUP VARIABLES")
                    #IF RUNNING SCRIPT LOCALLY 
    filenum = 0
    component = [2]
    station=  "IWEH"
    start_day = 31
    end_day = 31.25
    overlap = 20
    duration = 40
    norm= "max"
    out_to_wav = False
    
#filenum = 2 


#print(filenum, component, station); sys.exit()
sample_rate = 100  # Hz
datapoints = 2.7E8
samples_per_day = 3600*24*sample_rate
random.seed(812) #makes data reproducable
#############################################GATHER RELEVANT DATA ###############################################################
"""
I will use the first ten mins of every hour for the station IWEH for the 15 days preceding earthquake
ev0000364000. 60000 samples = 10 minutes

"""

###############################################SPLIT DATA INTO CHUNKS#####################################
def split(X, chunk): 
    #print("good")
    global  x_ret
    
    #x_ret = np.array(ob)
    #global X
    global data_points_per_chunk
    data_points_per_chunk = int(len(X)/chunk)
    #print(data_points_per_chunk, len(X), chunk, gap, len(X)/gap)    
    assert len(stationname) >= 1, "THERE MUST BE AT LEAST ONE STATION"
    assert  gap % data_points_per_chunk == 0, "CHUNKS MUST BE MULTIPLE OF NUMBER STATIONS"
    """
    Ensuring the number of data points per chunk is 
    a multiple of the number of points per stations avoids overlap 
    as each chunk will belong to one station only 
    """
    if len(stationname) == 1:
        x_ret = np.empty((chunk,data_points_per_chunk+1)) #file for chunks
        for i in range(0, chunk): #ITERATE DATA AND SPLIT INTO CHUNKS 
            x_ret[i,0] = i*data_points_per_chunk #UPLOADS TIME STAMP FOR BEGINNING OF TIME SERIES
        
            add = np.reshape(X[i*data_points_per_chunk:(i+1)*data_points_per_chunk], data_points_per_chunk)
            #print(add.shape); sys.exit()
            x_ret[i, 1:] = add 
            #I NEED TO UPLOAD EACH OF XS POSITIION IN TIME HERE 
    else: ## IF THERE IS MORE THAN ONE STATION 
        global n, station_loc
        n = int(gap / data_points_per_chunk) #numbers chunks per station
        station_loc = 0 
        x_ret = np.empty((chunk,data_points_per_chunk+2)) #ADDITIONAL BECAUSE WE HAVE MORE STATIONS 
        loop = 0 
        for i in range(0, chunk): #ITERATE DATA AND SPLIT INTO CHUNKS 
            x_ret[i,0] = (i- loop*n +i%n)*data_points_per_chunk #UPLOADS TIME STAMP FOR BEGINNING OF TIME SERIES
            if i % n == 0 and i != 0:
                station_loc +=1 #starts at 0, assumes equal number data points per station
                loop += 1 
            #if station_loc - int((i+1)*(data_points_per_chunk//gap)) >1:
            x_ret[i,1] = station_loc
            add = np.reshape(X[i*data_points_per_chunk:(i+1)*data_points_per_chunk], data_points_per_chunk)
            x_ret[i, 2:] = add
            
            #print(add, station_loc)
    
    X_train, X_test = train_test_split(
        x_ret, test_size=0.2, shuffle=True, random_state=812) #SPLITS THE TRAINING AND TESTING CHUNKS UP 
    #sys.exit("Successfully uploaded")
    del x_ret  # need to test what effect this has on the dataset.
    return X_train, X_test


def windows(X, duration, overlap): #WITHIN TRAINING AND TESTING CHUNKS SPLITS THESE INTO SMALLER WINDOWS
    #print("Good")
    X_return = []
    # assumes all the chunks are of equal length , puts it into units of seconds
    global length
    length = int(len(X[0])/100) #divide by sample rate length of chunks in seconds 
    assert len(stationname) >= 1, "THERE MUST BE AT LEAST ONE STATION"
    if len(stationname) == 1: 
        for u in X:  # for each chunk in x val or x train
            #X_chunk = []
            #print(u)
            #global index
            index = u[0] #extract u's location in time (INDEX UNITS0) 
            
            #sys.exit()
            for i in range(0, length - (duration+1), duration - overlap):  # less overlap = bigger jump
                # converts seconds into sameples
                spec_pos = i*100+index #spectrogram position in time
                tobe = [spec_pos] #the first number of the datafile corresponds to its position in time 
                add = u[(i+1)*100: (i+1+duration)*100]
                add = normalize([add], norm = norm) #normalise requires 2d shapoe
                add = np.reshape (add, (duration*100,)) #reshape it back to original
                tobe.append(add) #time stamp at the beginnign of each file
                
                #print(tobe); sys.exit()
                X_return.append(tobe)
                #plt.plot(u[i*100: (i+duration)*100]); sys.exit()
            # X_return.append(X_chunk) #appends all the window for each chunk
    else: 
        for u in X:  # for each chunk in x val or x train
            #X_chunk = []
            #print(u)
            #global index
            index = u[0] #extract u's location in time (INDEX UNITS0) 
            station = u[1]
            #sys.exit()
            for i in range(0, length - (duration+1), duration - overlap):  # less overlap = bigger jump
                # converts seconds into sameples
                spec_pos = i*100+index #spectrogram position in time
                tobe = [spec_pos, station] #the first number of the datafile corresponds to its position in time 
                add = u[(i+1)*100: (i+1+duration)*100]
                #print(add)
                #print(add.shape)
                add = normalize([add], norm = norm) #normalise requires 2d shapoe
                #print(add); sys.exit()
                add = np.reshape (add, (4000,)) #reshape it back to original
                tobe.append(add) #time stamp at the beginnign of each file
                
                #print(tobe); sys.exit()
                X_return.append(tobe)
                #plt.plot(u[i*100: (i+duration)*100]); sys.exit()
            # X_return.append(X_chunk) #appends all the window for each chunk
        
    del X
    #X_return = np.asarray(X_return) 
    return X_return  # all chunks lumped togehter

#------------------------------------------------------------------------------------------------------------------
def get_metadata(file, station_name):
    stations = file.keys()
    #global attributes
    attributes = np.empty((len(stations), 4))
    for i, x in enumerate(stations):
        attributes[i, :] = [float(file[x].attrs[u]) for u in [i for i in file[x].attrs.keys()]]
        #global time 
    mag,time = [i[1] for i in f.attrs.items() if i[0] in ["time", "mag"]]
    attribute_names = [r for r in f[station_name].attrs.keys()]
    attribute_names.append("Epicentral time")
    
    #attributes[:,0] = attribute_names
    return attribute_names, attributes, time, mag

#-----------------------------------------------------------------------------------------------------------------
if station == "all": 
    with h5py.File(directory + files[filenum]) as f:
        stationname = list(f.keys())
else:
    stationname = [station]
#--------------------------------------------------------------------------------------------------------------------  
master_file = np.zeros((len(component),len(stationname)*(int(end_day*samples_per_day) - int(start_day*samples_per_day))))#; sys.exit() #all earthquake data in here!
with h5py.File(directory + files[filenum]) as f:
    print(f.keys())
    attribute_names, attributes, time, mag = get_metadata(f, stationname[0]) #ASSIGN ATTRIBUTES TO EACH SPECTRORGRAM!
    #print(attribute_names, attributes)
    include_labels = ["Distance from Epicentre", "Elevation (sea level)", "Latitude", "Longitude"]
    stations = [i for i in f.keys()]
    filtered_attribute_names = [label for label in include_labels if label in attribute_names]
    df = pd.DataFrame({label: attributes[stations.index(label)] for label in stations})
    #df.index(include_labels)
    #df = pd.DataFrame(attributes)
    df.insert(0,"Time", time); df.insert(0,"mag", mag) 
    
    df.insert(0, "Info", include_labels)
    #df = pd.DataFrame(data)#.set_index("station")
    df.to_csv(f"/nobackup/vsbh19/Earthquake_Meta/attributes_of_{files[filenum][:-3]}.csv")
    #print([i for i in f.attrs.items()])
    #X = f.get(station)[:]
    #sys.exit() 
    gap = (end_day-start_day)*samples_per_day
    for i,u in enumerate(stationname): 
        #print(i,u)
        X = np.array(f[u][component]) #30 DAY TIMESERIES FOR STATION AND COMPONENT
        #print(end_day*samples_per_day)
        X = X[:,int(start_day*samples_per_day):int(end_day*samples_per_day)] #filters data for number of days I want to look at
        #print(len(X))
        #print(X.shape)
        master_file[:,int(i*gap):int((i+1)*gap)] = X
        
    if out_to_wav == True:
        print("GENERATING RAW DATA FILE")
        with h5py.File(f"/nobackup/vsbh19/wav_files/RawData_{files[filenum][:-3]}_{station}_{component}_days{start_day}-{end_day}.h5", "w") as w:
            w.create_dataset("data", data = master_file)
            sys.exit("RAW DATAFILE GENERATED") #exit as I only want the full timeseries for wav file
    
    # plt.plot(X[0:4000])
    # sys.exit()
    # IWEH : on coastline
    # KI2H AND MKJH ARE RURAL LOCTION
    # MNEH are located Namie
    
     
master_file = np.reshape(master_file, (len(master_file[0,:]),len(master_file)))
#SWITCH MASTER FILE ROUND TO AVOID PROPOGATION OF ERROR
chunk_num  = int(100*len(master_file)/gap)
X_train, X_test = split(master_file, chunk_num)  # 400*length statiion wanted chunks wanted
#sys.exit("Done")
del master_file  # save some memeory
#sys.exit()
#X_train = np.reshae(X_train)
# 20 seconds duration 19 seconds overlap
X_train = windows(X_train, duration, overlap)
X_test = windows(X_test, duration, overlap)
#sys.exit("Done")
#number_samples_window = np.asarray(X_val).shape[1]
#print("Number samples", len(X_train))


# seg = int(700 *(duration/40)) #calculates time windows in proportion to spectrogram
# lap = int(684 *(duration/40)) 
# seg = 1200
# lap = 1193
# seg = 600
# lap = 517 #at 19,01
#seg = 1200 #LENGTH OF SEGMENT 
#lap = 1131 #AMOUNT OF OVERLAP 
seg =720*(duration/40) #for SHAPE=(None, 140, 41, 1)
lap =639*(duration/40)#from the snover study of 90s overlap
#seg = 12000  #for 11.5 hertz and 140 frequency bins 
#seg = 1187 #WHAT WAS UNTIL 19 01
beta = np.pi*1.8 #beta = pi*alpha (used in Snover (2020))
def spectrogram_gen(X, seg, lap, beta):
    if len(stationname) == 1:
        frequencies, times, Sxx = spectrogram(X[0][1], 
                                  100, nperseg=seg, 
                                  noverlap=lap, 
                                  window=("kaiser",  beta))#; sys.exit()
        second = False

    else: 
        frequencies, times, Sxx = spectrogram(X[0][2], 
                                      100, nperseg=seg, 
                                      noverlap=lap, 
                                      window=("kaiser",  beta))#; sys.exit()
        second = True
    # TASK INCEASE WINDOW LENGTH TO GET BETTER REPRESENTATION OF LOWER FREQUENCY
    fre_cro_i = 140 #was 140  # indexes the data so that it is CROPPED to 45Hz (originally), now at 24 hertz
    
    Sxx = Sxx[:fre_cro_i]
    frequencies = frequencies[:fre_cro_i]
    #print(times, Sxx); sys.exit()
    # fig = plt.figure(figsize=(30, 10))
    # fig.add_subplot(1, 2, 1)
    # plt.pcolormesh(times, frequencies, Sxx)
    # plt.colorbar()
    # # ax[0].suptitle(f"{station}")
    #fig.add_subplot(3, 2, 1)
    
    # plt.plot(X_train[0][1])
    #sys.exit()
    
    #plt.close(fig)
    tobe = np.zeros((len(X), len(frequencies), len(times)))
    tobe[0, :, :] = Sxx
    # ^initial run to get sizes of frequencies and time
    indices = [X[0][0]]#; sys.exit()
    if second == False: 
        for i, u in enumerate(X[1:]):
        # print(u)
        #print(u[1], len(u)); sys.exit()
            indices.append(u[0])
            
            frequencies, times, Sxx = spectrogram(u[1], 100,
                                              nperseg=seg, noverlap=lap,
                                              window=("kaiser",  beta))
            Sxx = Sxx[:fre_cro_i]
            frequencies = frequencies[:fre_cro_i]
            #sys.exit()
            tobe[i, :, :] = Sxx
        # print(Sxx)
        stations_indices = [None]
    else: 
        stations_indices = [X[0][1]]
        for i, u in enumerate(X[1:]):
            #print("GOOD")
            #print(u[1], len(u)); sys.exit()
            indices.append(u[0])
            stations_indices.append(u[1])
            frequencies, times, Sxx = spectrogram(u[2], 100,
                                              nperseg=seg, noverlap=lap,
                                              window=("kaiser",  beta))
            Sxx = Sxx[:fre_cro_i]
            frequencies = frequencies[:fre_cro_i]
            #sys.exit()
            tobe[i, :, :] = Sxx
            # print(Sxx)
            # sys.exit()
            #print(tobe.shape); sys.exit()
    return  tobe, frequencies, times, indices, stations_indices, second



#-------------------------------------------------------------------------------------------------------------

tobe_train, frequencies_train, times_train, indices_train, stations_indices_train, second = spectrogram_gen(X_train, 
                                                                                                         seg, lap, beta)

#------------------------------UPLOADING TRAINING AND VALIDATION DATA-----------------------------------
with h5py.File(f"{directory}Spectrograms_" + files[filenum][:-3] + "_" + str(station) + "_" + str(component)+ "_" + str(duration) + "_" + str(norm) + "_training.h5", "w") as spectrograms:
    
    # times *= 10 # converts times to seconds
    #h5py._errors.unsilence_errors() #if errornous h5 file
    indices = np.array(indices_train)
    plt.plot(indices); sys.exit()
    dset = spectrograms.create_dataset("Data", data=tobe_train)
    locset = spectrograms.create_dataset("Indices", data = indices)
    #-------------------for lots of stations------------------------------
    if second == True: 
        spectrograms.create_dataset("Stations", data = stations_indices_train)
        print("Uploading multiple stations onto spectrogram TRAINING file")
        dset.attrs["Station"] = "ALL"
    #-------------------for one station----------------------------
    else: 
        spectrograms.create_dataset("Stations", data = stations.index(station))
        dset.attrs["Station"] = station
    spectrograms["Frequency scale"] = np.array(frequencies_train)
    spectrograms["Frequency scale"].make_scale("Frequency scale")
    spectrograms["Data"].dims[1].attach_scale(spectrograms["Frequency scale"])
    spectrograms["Time scale"] = np.array(times_train)
    spectrograms["Time scale"].make_scale("Time scale")
    spectrograms["Data"].dims[0].attach_scale(spectrograms["Time scale"])
    
    #spectrograms.create_dataset("frequencies", data = frequencies)
    #spectrograms.create_dataset("times", data = times)
    #spectrograms.create_dataset("number_samples_window", data = numb,queueer_samples_window)
    # split data first
    #X = preprocessing.normalize([X] , norm= "l2")[0]
    print("Successfully uploaded ", len(X_train), " training samples")
    spectrograms.flush()

#-------------------------------------UPLOAD TESTING DATA----------------------------------------------
del tobe_train, frequencies_train, times_train, indices_train, stations_indices_train, second #SAVE MEMORY

tobe_test, frequencies_test, times_test, indices_test, stations_indices_test, second = spectrogram_gen(X_test, 
                                                                                                    seg, lap, beta)

with h5py.File(f"{directory}TESTING_Spectrograms_{files[filenum][:-3]}_{station}_{component}_{duration}_{norm}.h5", "w") as spectrograms:
    indices = np.array(indices_test)
    dset = spectrograms.create_dataset("Data", data=tobe_test)
    locset = spectrograms.create_dataset("Indices", data = indices)
    #-------------------for lots of stations------------------------------
    if second == True: 
        spectrograms.create_dataset("Stations", data = stations_indices_test)
        print("Uploading multiple stations onto spectrogram TESTING file")
        dset.attrs["Station"] = "ALL"
    #-------------------for one station----------------------------
    else: 
        spectrograms.create_dataset("Stations", data = stations.index(station))
        dset.attrs["Station"] = station
    spectrograms["Frequency scale"] = np.array(frequencies_test)
    spectrograms["Frequency scale"].make_scale("Frequency scale")
    spectrograms["Data"].dims[1].attach_scale(spectrograms["Frequency scale"])
    spectrograms["Time scale"] = np.array(times_test)
    spectrograms["Time scale"].make_scale("Time scale")
    spectrograms["Data"].dims[0].attach_scale(spectrograms["Time scale"])
    print("Successfully uploaded ", len(X_test), " testing samples")
    spectrograms.flush()
sys.exit(1)
