#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:08:16 2024

@author: vsbh19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:37:37 2024

@author: vsbh19
"""
#-------------------------------------------modules loasdd
import matplotlib.pyplot as pltN
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import random
import numpy as np
import h5py
import sys
import os
import pandas as pd
from retrieve_earthquake_prelim import heythem
from obspy import UTCDateTime
import csv 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
from scipy.signal import savgol_filter, correlate, square, correlation_lags, butter, filtfilt, freqs
from drumplot import drumplot
import matplotlib.pyplot as plt
from  butter_filter import butter_filter
from geometric_analysis import Geometric_analysis as gma
files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5","ev0000447288.h5", "ev0000734973.h5", "Sinosoid.h5"]
file_times = ["2014-07-11T19:22:00", "2016-04-14T12:26:35", "2021-03-20T09:09:44", "2022-01-21T16:08:37", "2016-11-21T20:59:46", "2015-02-16T23:06:34.680000", "2000-01-01T00:00.0000", "1900-01-01T00:00:00.0000"]
file_times = [UTCDateTime(i) for i in file_times]#; sys.exit()
date_name = "Jan24"
try:
    filenum = int(sys.argv[1]) # Which file we want to read.
    component = [int(sys.argv[3])] #0,1,2 = X,Y,Z
    stationname = sys.argv[2] #station we want to read
    duration = int(sys.argv[4])
    n_clusters = int(sys.argv[5])
    start_day = float(sys.argv[6])
    end_day = float(sys.argv[7])
    norm = sys.argv[8]
    switch = bool(sys.argv[9])
    testing = int(sys.argv[10]) 
    latent_dim = int(sys.argv[11])
except:
    filenum =5
    component = [2] #"U"
    stationname = "all"
    duration = 120
    start_day =0
    end_day = 31.25
    norm = "l2"
    n_clusters = 6
    switch = "False"
    testing = 0
    high_pass = 1
    latent_dim = 15
noise_stamp = False
drum_query = False
###plots preference

produce_stack = False
individual_corr_plots = True
summ_corr_plots = True
sample_rate = 100
samples_per_day = 60*60*24*sample_rate #seconds*mins*hours*sample rate in hertz
month_diff = 2700000 # seconds month
day_diff = 3600*24 #seconds in day
minmag = 3
#TESTING = True 
print(f"************TESTING: {testing}*****************")

#%%

path_add_cluster = f"{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}_Lat{latent_dim}_C{n_clusters}"
#path_out_cluster = f"%s%s%s{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}" %("TESTING" if testing == 1 else "", "FLIP" if switch == True else "", {endtt.year, endtt.month}if alt_month == True else "")
#path_add_cluster = f"{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}_C{n_clusters}"
#path_add_cluster = f"{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}"
'''for the tremor dataset'''
#path_add_cluster = f"MSeedSpectrograms_April2005_{stationname}_{component}_{duration}_{norm}_C{n_clusters}"

#%% low pass filter 



#%%
directory = os.path.join("/nobackup/vsbh19/h5_files/")
with h5py.File(directory + files[filenum]) as f:
    stations = [i for i in f.keys()]
station_list = np.arange(0, len(stations), dtype = int)
colours = ["red", "blue", "green", "purple",  "orange", "black", "cyan", "violet"] #universal coloring system to stop man getting confused
if files[filenum] == "Sinosoid.h5":
    synthetic = True #we dont want to dry and drag out events for synthetic data!
else:
    synthetic = False

nobackupname = os.path.dirname("/nobackup/vsbh19/snovermodels/")

with h5py.File(nobackupname + f'/%s%sDEC_Training_LatentSpaceData_{path_add_cluster}.h5' %("FLIP" if switch == "True" else "", "NOISE_STA" if noise_stamp else ""), 'r') as nf:
    print(nf.keys())#; sys.exit()
    #val_reconst = nf.get("Val_ReconstructData")[:]
    #val_enc = nf.get("Val_EncodedData")[:]
    #enc_train =nf.get("Train_EncodedData")[:]
    train_reconst = nf.get("Train_ReconstructData")[:]
    labels_train = nf.get("Labels_Train")[:]
    try:
        labels_test = nf.get("Labels_Test")[:]
    #train_enc = nf.get("Train_Reconst")[:]
    #train_reconst = nf.get("Train_ReconstructData")[:]
        test_reconst = nf.get("Test_ReconstructData")[:]
    except: 
        print("No testing data in Training Latent Space data")

    

with h5py.File(f"/nobackup/vsbh19/training_datasets/%s%sX_train_X_val_{path_add_cluster}.h5" %("FLIP" if switch == "True" else "", "NOISE_STA" if noise_stamp else ""), "r") as f:
    print(f.keys())#; sys.exit()
    X_val = f.get("X_val")[:]
    X_train = f.get("X_train")[:]
    try:
        fre_scale = f.get("Frequency scale")[:]
        time_scale = f.get("Time scale")[:]
    except: 
        print("No frequency scale or time scale")
    train_enc = f.get("Trained_Encoded_Data")[:]
    try:
        X_train_pos = f.get("X_train_pos")[:]
        X_val_pos = f.get("X_val_pos")[:]
    except: 
        print("TIME DATA NOT INCLUDED")
    ######################################   TESTING DATA ###################################
    try:
        X_test_labels = f.get("Labels_Test")[:]
        test_enc = f.get("Test_Encoded_Data")[:]
    except: 
        print("No testing data in X train X val")
    ######################################## STATIONS ############################
    try:
        X_train_station = f.get("X_train_station")[:]
        X_val_station = f.get("X_val_station")[:]
        more_stat = True
    except: 
         print("Singular station used")
         more_stat = False
    #-----------------SYNTHETIC DATA + SUPERVISED.UNSUPERVISED---------------------
    try: 
        truths_train = f["Truths_Train"][:]
        truths_val = f["Truths_Val"][:]
        supervised = True
        
    except: 
        supervised = False
        print("No truth data UNSUPERVISED MODEL")
#############################PLOTTING OF DIFFERENT CLUSTERS###################################################

if testing == 1: 
    with h5py.File(f"/nobackup/vsbh19/training_datasets/X_TEST_{path_add_cluster}.h5", "r") as f2:
        print(f2.keys())
        X_test = f2.get("X_test")[:]
        X_test_pos = f2.get("X_test_pos")[:]
        X_test_station = f2.get("X_test_station")[:]
        test_enc = f2.get("Test_Encoded_Data")[:]
        


#%%
if drum_query == True: 
    station_num = 0
    start_day_drum = 0
    end_day_drum = 31.25
    with h5py.File(f"/nobackup/vsbh19/wav_files/RawData_{files[filenum][:-3]}_{stations[station_num]}_{component}_days0-31.25.h5")as r:
        
        global drum_array; drum_array = np.asarray(r["data"])
        drum_array = drum_array[0, int(start_day_drum*day_diff*sample_rate):int(end_day_drum*day_diff*sample_rate)]
    #sys.exit()
def print_cluster_size(labels):
    """
    Shows the number of samples assigned to each cluster. 
    # Example 
    ```
        print_cluster_size(labels=kmeans_labels)
    ```
    # Arguments
        labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster 
                that the samples in the clustered data set (with the same index) was assigned to. Array must be the same length as
                data.shape[0]. where 'data' is the clustered data set. 
    """
    num_labels = max(labels) + 1
    for j in range(0,num_labels):
        label_idx = np.where(labels==j)[0]
        print("Label " + str(j) + ": " + str(label_idx.shape[0]))
#%%
def day_night(labels, time_func):
    end_points = np.max(time_func)
    #print(end_points)
    end_time = UTCDateTime(file_times[filenum])
    hour = int(str(end_time.time)[:2])
    d_sta = 8 #hours
    d_end = 20 #hours
    global phase_shift
    if hour <= d_sta or hour >= d_end: 
        night  = 1 
    else: 
        night = 0 
    if night == 1: 
        dnight = hour + (24-d_end) if hour <= d_sta else hour  - d_end
        #print(dnight)
        dnight *= 3600
        hook = end_points - dnight
        #print(hook, end_points, end_points - hook)
        #ax33.axvspan(hook, end_points, color = "grey")
        phase_shift = 1*(end_points - hook) 
        end = -1*np.ones((int(end_points - hook))) #end of box car 
    else: 
        dday = hour - d_sta
        dday *= 3600
        hook = end_points - dday 
        #ax33.axvspan(hook, end_points, color = "white")
        phase_shift = -1*(end_points - hook) 
        end = np.ones((int(end_points - hook)))
    dcol = ["white", "grey"]
    day_range = range(0, int(hook//(day_diff*0.5)))
    #print(end_points - hook, day_range)
    for f in day_range:
        day = f % 2
        #ax33.axvspan(hook - 0.5*day_diff, hook, color = dcol[day])
        #print(hook - 0.5*day_diff, hook, day_diff * 0.5)
        hook -= day_diff*0.5
    global time_total; time_total = np.linspace(0,np.max(time_func), len(labels[0][1])) #takes into consideration there will be gapes in the actual time
    
    fre_sq = 2*np.pi*(1/day_diff)
    #global phased_period; phased_period = [i+phase_shift for i in period*time_total]
    #global pwm; pwm = square(fre_sq*time_total +phase_shift , duty = 0.5)
    global pwm; pwm = -1*np.sin(fre_sq*time_total*3600*24 + phase_shift)
    # print(time_total)
    # plt.plot(pwm); sys.exit()
    return pwm
#%%
def  Pearson_correlation(X2,Y2):
    """
    Parameters
    ----------
    X : 1d Numpy Array 
        Variable X
    Y : 1d Numpy Array
        Variable Y

    Returns
    -------
    X: Normalised array 
    Y: Normalised array 
    Pearson correlation moment 
    """
    
    X=X2
    Y=Y2
    #ensure same shape and numpy array type 
    
    assert isinstance(X, np.ndarray), "X MUST BE A NUMPY ARRAY"
    assert isinstance(Y, np.ndarray), "Y MUST A NUMPY ARRAY"
    
    length_want = max(Y.shape[0], X.shape[0])
    if len(X) < len(Y):
        X = np.interp(np.linspace(0,len(X), length_want), np.arange(0,len(X)), X )
    elif len(X) > len(Y):
        Y = np.interp(np.linspace(0,len(Y), length_want), np.arange(0,len(Y)), Y )
    
    #remove mean and normalise 
    #if np.min(X) > 0:#we dont want to normalise the noise
    X3 =  X  - np.mean(X) #MAJOR ISSUE HERE 17,05 REASSIGNING THE VARIABLE INTO MEMORY 
    X3 /= np.max(abs(X))
    #if np.min(Y) > 0: #ditto
    Y3 = Y - np.mean(Y) #MAJOR ISSUE HERE 17,05 
    Y3 /= np.max(abs(Y))
    stds = [np.std(i) for i in [X3,Y3]] #standard deviation
    cov = (1/X.shape[0]) * np.dot(X3,Y3) #covariance of the sum of the multiplication of series
    #divided by the length of the arrays (number sampels)
    pearson = cov/(stds[0]*stds[1])
    return X3,Y3, pearson 

#%%

def noise_analysis(labels, truths, indices):
    """ 
    Plots up metrics of accuracy precision and recall against noise metrics"""
    label_des = ["Nothing", "Wavelet", "Sinosoids", "Frequency modulation"]
    unique_indices = sorted(set(indices))
    unique_labels = sorted(set(labels))
    print(labels, truths)
    f, axes =  plt.subplots(len(unique_labels),1,figsize = (10,15))
    for u in unique_labels: #one fig graph per label
        
        accuracyl = []; precisionl = []
        truth_loc = np.where(truths == u)#where are the labels for each truth
        print(truth_loc)
        label_loc = [labels[i] for i in truth_loc][0]#find what the label was actually assigned
        
        #label_truths = 
        noise_loc = [indices[i] for i in truth_loc][0] #gather all the noises for consistuent truth u
        for i in unique_indices: #for each noise level we will calculate accuracy and precision 
            noise_loc_loc = np.where(noise_loc == i) #locations for distinct noise i
            label_loc_loc = [label_loc[i] for i in noise_loc_loc][0] #get the labels
            print(label_loc_loc)
            dummy_u = u*np.ones(len(label_loc_loc)) #such that we have an array of truths
            accuracy = accuracy_score(dummy_u, label_loc_loc)
            #try: 
            precision = precision_score(dummy_u, label_loc_loc, average = "macro")
            #except: #div 0 error
            #precision = 0
            accuracyl.append(accuracy); precisionl.append(precision)
        
        axes[u].plot(unique_indices, accuracyl, color = colours[u], label = "Acc")#unique_indices, precisionl)
        axes[u].plot(unique_indices, precisionl, color = colours[u], label = "Prec", linestyle ="--")
        axes[u].set_xscale("log")
        axes[u].axvline(5, linestyle = "--", color  = "lightgrey", label = "SNR Lower Bound")
        axes[u].axvline(3.5, linestyle = "--", color  = "darkgrey", label = "SNR Upper Bound")
        axes[u].set_title(f"Propogation for LABEL{u} : {label_des[u]}")
        axes[u].legend()
    """
    SNR ranges from 0.20 to 0.28 such that the NSR ranges 
    from 5 to 3.5 """
    f.suptitle(f"Accuracy propogation for all labels")
    f.supxlabel("Logarithmic noise factor")
    plt.savefig("/home/vsbh19/plots/Clusters_station/Sinosoid.h5/NoiseMetrics.png")
    #sys.exit()
#%% FUNCTION TO PRINT ALL CLUSTERS
def print_all_clusters(data, labels, num_clusters, positions, stations, **kwargs):
    """
    Shows six examples of spectrograms assigned to each cluster. 
    # Example 
    ```
        print_all_clusters(data=X_train, labels=kmeans_labels, num_clusters=10)
    ```
    # Arguments
        data: data set (4th rank tensor) that was used as the input for the clustering algorithm used.
        labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster 
                that the samples in 'data' (with the same index) was assigned to. Array must be the same length as
                data.shape[0].
        num_clusters: The number of clusters that the data was seperated in to.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    if "truths" in kwargs:
        truths = kwargs["truths"]
    plot_time_dist = kwargs["plot_time_dist"] if "plot_time_dist" in kwargs else False
    plot_clusters_only = kwargs.get("plot_clusters_only", False)
    fig1=plt.figure() 
    global time_idxes, station_idxes, fre
    global clustered_data
    clustered_data = pd.DataFrame(columns=["True ID", "Label", "Spectrogram", "Time", "Station"])
    amounts = []
    try: 
        kwargs.get("alt_months")
    except: 
        alt_months = 0 
    
    ###
    #global y_limits
    global current
    for cluster_num in range(0,num_clusters):
        fig = plt.figure(figsize=(14,5))
        num_labels = max(labels) + 1
        cnt = 0
        num_fig = 12; ration=2
        global  time_idx, station_idx, spec_num
        label_idx = np.where(labels==cluster_num)[0] #where are the labels located?
        #idx_in = label_idx
        try:
            cluster_array = [cluster_num for i in label_idx]
            time_idx = [positions[i] for i in label_idx] #where are the times for this cluster?
            station_idx = [stations[i] for i in label_idx] #where are the stations for this cluster?
            spec_num = [i for i in range(0, len(label_idx))]
        except: 
            print("error in creating dataset")
        amount = len(label_idx)
        amounts.append(amount)
        #time_idxes.append(time_idx)
        #station_idxes.append(station_idx)
        global to_dataframe
        try:
            to_dataframe = pd.DataFrame({
                            "True ID": label_idx,
                            "Label": cluster_array,
                            "Spectrogram": spec_num, 
                            "Time": [i/samples_per_day + start_day for i in time_idx], #this converts time samples to days
                            "Station": station_idx})
            clustered_data = pd.concat([clustered_data, to_dataframe], ignore_index=True)
            clustered_data.to_csv("out.csv")
        except: 
            print("Error in creating pandas dataframe")
        gs = GridSpec(ration,int(num_fig/ration),width_ratios = [1 for i in range(int(num_fig/ration))])
        if len(label_idx) < num_fig:
            for i in range(len(label_idx)):
                ax = plt.subplot(gs[i])
                plot_data = data[label_idx[i],y_limits[0]:y_limits[1],:]; print(plot_data.shape)#; sys.exit()
                ax.imshow(np.reshape(plot_data, (y_limit_difference,41)))
                ax.set_ylabel('Frequency (Hz)')
                
                ax.set_xticks(secs_pos, secs)
                ax.set_xlabel('Time(s)')
                ax.set_yticks(fre_pos, fre)
                ax.set_aspect(1)
                #ax.set_ylim([0,140])
                ax.invert_yaxis()
                #plt.colorbar()
        else: 
            for i in range(0,num_fig):
                cnt = cnt + 1
                ax = plt.subplot(gs[i])
                plot_data = data[label_idx[i],y_limits[0]:y_limits[1],:]; #print(plot_data.shape); sys.exit()
                ax.imshow(np.reshape(plot_data, (y_limit_difference,41)))
                ax.set_ylabel('Frequency (Hz)')
                
                ax.set_xlabel('Time(s)')
                ax.set_yticks(fre_pos, fre)
                ax.set_xticks(secs_pos, secs)
                ax.set_aspect(1)
                #ax.set_ylim([0,140])
                ax.invert_yaxis()
                #plt.colorbar()
            plt.suptitle(f"Label {cluster_num} \t #:{len(label_idx)} \t %:{int(100*len(label_idx)/len(data))}", ha='left', va='center', fontsize=28) 
            plt.savefig(f"/home/vsbh19/plots/Clusters/%s{path_add_cluster[:-2]}{cluster_num}.png" %("FLIP" if switch == True else ""))
        
        plt.tight_layout()
    
    #%% ----DRUM PLOT ----------------------------
    if plot_clusters_only:
        return "Done"
    
    colours = ["red", "blue", "green", "purple",  "orange", "black", "cyan", "violet"]#; sys.exit()
    
    #global timess; timess = clustered_data["Time"]#; sys.exit()
    if plot_time_dist == True: 
        for i in range(n_clusters):
            global timess; timess = clustered_data.loc[clustered_data['Station'] == i, 'Time']
            timess = timess.sort_values(ascending = True)
            counts, bins = np.histogram([timess], bins = np.arange(0,31.25,0.001333))
            
            plt.bar(bins[:-1], counts, width = np.diff(bins))
            plt.save(f"Distribution for station {i}")
            plt.show()
    if drum_query == True:
        file = files[filenum]
        plots = drumplot(drum_array,clustered_data, duration, n_clusters, 
                     start_day_drum, end_day_drum, station_num, file,  path_add_cluster, high_pass,
                     end_of_series_time = file_times[filenum], 
                     clipping = True, colors = colours, amount = 10, nrows=8, typeplot = "drumplot", spec_length = duration,
                     Original_Spectrograms = data, Reconst_Spectrograms = kwargs.get("reconst"),
                     timebounds = (np.min(time_scale),np.max(time_scale))); 
        plots.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/%s%sSERIESCLUSTERS_{station_num}_{component}_{duration}_{norm}_Cluster{n_clusters}_Examples.png" %("TESTING" if testing == 1 else "", "FLIP" if switch == True else ""))
        plt.clf()
        sys.exit("DRUMPLOTS CREATED")
        
     
    
    #%% -------PLOT LABELS IN TIME AND SPACE-----------------------
    plt.show()
    #sys.exit()
    plt.figure(2)
    global smo
    step = (end_day-start_day)/50 #in days !!
    sliding_step = step/16
    global steps, values_y, r, reader, steps_sec, event_mag, event_times, reader, mag5
    steps = np.linspace(start_day, end_day, num = int((end_day-start_day)/step)) #UNIT: DAYS
    #steps = np.linspace(start_day, end_day, num = int((end_day-start_day) - step /sliding_step)) #UNIT: DAYS
    #%% CONFUSION MATRIX FOR  SUPVERVISED AND SYNTHETIC LEARNING 
    if supervised == True:  
        try: 
            truths 
        except: 
            sys.exit("YOU NEED TO DEFINE WHICH TRUTHS YOURE USING AS A KWARG truths =")
        cm = confusion_matrix(truths, labels)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/Sinosoid.h5/Confusion_C{n_clusters}_L{max(truths)}.png")
        noise_analysis(labels, truths, positions)
        # print("ACCRUACY", accuracy_score(truths, labels))
        # print("PRECISION", precision_score(truths, labels))
        
      #%% READ EVENTS 
    try:
        with open(f"eventt_{str(endtt - month_diff)[:7]}_to_{str(endtt)[:7]}_mag{minmag}.txt","r") as r:    
    
            reader = pd.read_csv(r, delimiter='|', names=['ID', 'Time', "Location", "Magnitude", "Depths" ])
            mag_col = reader["Magnitude"]
            new_mag_col = mag_col.apply(lambda x : float(x[:-5])) #converts magnitude  coloumn to floast
            time_col = reader["Time"]
            new_time_col = time_col.apply(lambda x: endtt - UTCDateTime(x)) #times are now relative to end
            #PLEASE CHECK THE ORDER OF THE ABOVE AND MAKE SURE NEW TIME COL IS IN THE SAME ORDER AND STEPS SEC 
            reader["Magnitude"] = new_mag_col
            reader["Time"] = new_time_col 

    except FileNotFoundError:
        try:
            with open(f"events_{files[filenum]}_mag{float(minmag)}.txt","r") as r:
                reader = pd.read_csv(r, delimiter='|', names=['ID', 'Time', "Location", "Magnitude", "Depths" ])
        except FileNotFoundError: 
            print("NO EVENTS AVAILABLE")
            return
    global mag4, mag4list, mag4_sums, mag5_sums, mag3_sums
    
    #---------------------------------------plotting cluster frequencies for station-----------------------------------------------
    mag3 = reader.loc[reader["Magnitude"]>=3, "Time"]
    mag4 = reader.loc[reader["Magnitude"] >= 4, "Time"]
    #mag4_smoothe_rolling = mag4.rolling(3).apply(lambda x : len(x))
    mag5 = reader.loc[reader["Magnitude"]>=5, "Time"]
    steps_sec = [i*day_diff  for i in np.flip(steps)] #steps converted into days
    steps_sec_old = steps_sec
    mag3_sums = []; mag4_sums = []; mag5_sums = []
    for uu in steps_sec:
        mag3_contain, mag4_contain, mag5_contain = [v for v in mag3 if uu <= v <= uu+step*day_diff],[v for v in mag4 if uu <= v <= uu+step*day_diff],[v for v in mag5 if uu <= v <= uu+step*day_diff]
        mag3_sums.append(len(mag3_contain)),mag4_sums.append(len(mag4_contain)); mag5_sums.append(len(mag5_contain))
    
    window = 2 #window size for smoothing
    convolve_kernel = np.ones(window)/window
    
    mag4_sums_smoothe = np.convolve(mag4_sums, convolve_kernel, mode = "full") #returns the same output
    mag3_sums_smoothe = np.convolve(mag3_sums, convolve_kernel, mode = "full")
    mag5_sums_smoothe = np.convolve(mag5_sums, convolve_kernel, mode = "full")
    steps_new = [i*(len(mag4_sums_smoothe)/len(steps)) for i in steps]
    
    
    
    #plt.plot(steps_smooth, mag4_sums_smoothe, np.arange(0,len(mag4_sums)), mag4_sums)
             #np.arange(0, len(mag4_smoothe_rolling)), mag4_smoothe_rolling)
    #STEPS IS FLIPPED FOR A GIVEN ORDER AS TIME OF EVENTS INCREASES AWAY FROM EARTHQUAKE
    #sys.exit()
    #%% READ WEATHER 
    os.chdir("/nobackup/vsbh19/Earthquake_Meta")
    global weather_hourly; weather_hourly = pd.read_csv(f"weather{files[filenum][:-3]}hourly.csv")
    global weather; weather = pd.read_csv(f"weather{files[filenum][:-3]}.csv")
    
    moonphase = weather["moonphase"]
    moonphase = np.interp(np.linspace(0,len(moonphase), 768), np.arange(0,len(moonphase)), moonphase )
    weather_hourly = weather_hourly.loc[:,["windspeed", "sealevelpressure", "snowdepth"]]
    
    weather = weather_hourly
    weather["moonphase"] = moonphase
    global steps_smooth; steps_smooth = np.linspace(start_day,end_day,len(weather["moonphase"]))  #account for delay
    fr = np.random.normal(0,1,(steps_smooth.shape[0]))
    #fr_todataframe = pd.Series(fr, name = "Noise")
    
    fr= butter_filter(fr,"low", 1.5, 30, 1, False)
    weather["Noise"] = fr
    #%% CREATE DAY NIGHT
    
        
    #suspicous variables
    
    
    #%% PLOT LABEL OCCURANCES OVER TIME 
    
    def cnt(x):
        prev_count = 0
        for i in x:
            if i == lab:
                prev_count+=1
        return prev_count
    W = 100
    global sort, sort_fil, dist_y, dist_x, distances_y, distances_x
    distances_y , distances_x = [],[]
    global station_list
    
    fig, axes = plt.subplots(2,1,figsize = (10,15))
    global all_labels_all_stations; all_labels_all_stations = []
    for i in range(num_clusters):
        #length = len(time_idxes[i])
        length = amounts[i] 
        lab = i
        label_plot = int(i)*np.ones(shape = length) #linspace for label
        #time_idx = clustered_data.loc[clustered_data['Label'] == i, 'Time']
        time_idx_sorted = clustered_data.sort_values(by="Time", ascending = True)
        time_idx_sorted["Lab0Counts"] = time_idx_sorted["Label"].rolling(W,min_periods =1).apply(cnt)
        time_idx_sorted_2 = time_idx_sorted.groupby("Time", as_index = False)["Lab0Counts"].sum()
        data = time_idx_sorted_2.to_numpy()
        smo = savgol_filter(data[:,1],100,1,mode="nearest")
        #global time
        time = data[:,0]
        """ADJUST AS NEEDED"""
        all_labels_all_stations.append([time,smo])
        
    for i in range(0,2):
        axes[i].stackplot(time,all_labels_all_stations[0][1],all_labels_all_stations[1][1], all_labels_all_stations[2][1])# all_labels_all_stations[3][1], all_labels_all_stations[4][1], all_labels_all_stations[5][1])# all_labels[6][1], all_labels[7][1])
    
        axes[i].get_yaxis().set_ticks([0, W])
        axes[i].set_ylabel("Density")
        axes[i].set_xlabel('TIME IN DAYS')
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        global colors
        colors = colours
        axes[i].set_prop_cycle(color=colours)
    f, axes22 = plt.subplots(nrows=n_clusters, ncols=1)# sharey=True)
    # PLOT INVIDIUAL OCCURANCES OF EACH CLUSTER AT EACH STATION
    labels = np.arange(0,num_clusters)
    pearson_general = np.zeros((n_clusters, len(weather.columns)+4))
    # pwm = day_night(all_labels_all_stations, time)
    # pwm = [(r+1)*0.5*np.max(all_labels_all_stations[i][1]) for r in pwm] #shifts it above 0 np.max(all_labels[i][1])
    pwm = day_night(all_labels_all_stations, time)
    mag_compare = mag3_sums_smoothe; mag_num = 3 #CHANGE AT SAME TIME
    mag_compare =[mag3_sums_smoothe, mag4_sums_smoothe, mag5_sums_smoothe]
    #%% 
    """
    COMPARIRSON VARIABLE
    """
    compare_name = "snowdepth"
    global compare; compare = weather[compare_name]
    #compare = mag_compare
    f.suptitle(f"All labels all stations \n{path_add_cluster}")
    for i,r in enumerate(labels):
        
        pwm = [(r+1)*0.5*np.max(all_labels_all_stations[i][1]) for r in pwm] #shifts it above 0 np.max(all_labels[i][1])
        
        X,Y, *_ = Pearson_correlation(all_labels_all_stations[i][1], np.array(compare))
        ax33 = axes22.flat[i]
        axop33 = ax33.twinx()
        ax33.plot(all_labels_all_stations[i][0],all_labels_all_stations[i][1], label = str(i), color = colours[i])
        mag3_ratio = np.max(all_labels_all_stations[i][1])/np.max(mag3_sums)
        #mag3_x_vals = np.linspace(0,max(steps_sec), len(time))
        mag3_normal = [i*mag3_ratio for i in mag3_sums]
        #mag3_normal_interp = np.interp(mag3_x_vals, steps_sec, mag3_normal)
        #axop33.plot(np.linspace(0,31, len(mag3_normal)), mag3_normal)
    
        #print(i)
    #sys.exit()
        #ax.scatter(time_idx, label_plot, color = colours[i], s = 0.001)
        #ax33.plot(time, Y)
        
        #%% general stats summary irrespective of stations
        for index, (column_name, column_data) in enumerate(weather.items()):
        
            *_, pearson = Pearson_correlation(all_labels_all_stations[i][1], np.array(column_data))
            pearson_general[i,index+3] = pearson
        *_, pearson = Pearson_correlation(all_labels_all_stations[i][1], np.array(pwm))
        pearson_general[i, 0] = pearson
        for i7,u7 in enumerate(mag_compare):
            *_, pearson = Pearson_correlation(all_labels_all_stations[i][1], u7)
            pearson_general[i,i7+1] = pearson
    f.savefig(f"/home/vsbh19/plots/summary_tables/%s%s%s{path_add_cluster}_AllStations_AllLabels.png" %("TESTING" if testing == 1 else "", "FLIP" if switch == True else "", {endtt.year, endtt.month}if alt_month == True else ""))
    fig99, ax99 = plt.subplots(ncols=1, nrows=1)
    im = ax99.imshow(pearson_general, cmap = "seismic")
    plt.suptitle("ALL stations ALL labels")
    column_names = ["Day/Night", f"Mag{mag_num}eq" , "Mag4eq", "Mag5eq"]+ [i for i in weather.columns]
    ax99.set_xticks(np.arange(len(column_names)), column_names)
    cbar = fig99.colorbar(im)
    plt.setp(ax99.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    for i in range(pearson_general.shape[0]):
        for j in range(pearson_general.shape[1]):
            corrnum = str(pearson_general[i, j])
            # text = ax2.text(j, i, corrnum[1:4],
            #        ha="center", va="center", color="w", fontsize = "xx-small")
            text = ax99.text(j, i, corrnum[0:5],
                             ha="center", va="center", color="w", fontsize = "xx-small")
    #sys.exit()
    fig99.savefig(f"/home/vsbh19/plots/summary_tables/%s%s%s{path_add_cluster}_AllStations_AllLabels_Corr.png" %("TESTING" if testing == 1 else "", "FLIP" if switch == True else "", {endtt.year, endtt.month}if alt_month == True else ""))
        #%%
    dist_y = []
    dist_x =[]
    for z,u in enumerate(steps): #plots each occurance per 6 hours of day
        global values
        values_y = [v for v in time_idx if u <= v <= u+step]
        
        dist_y.append(len(values_y))#;print(len(values_y)); sys.exit()
    distances_y.append(dist_y)
    station_idx = clustered_data.loc[clustered_data['Label'] == i, 'Station']
    for c,q in enumerate(station_list):
        values_x = [v for v in station_idx if q-0.5 <= v <= q+0.5] #returns an array of all the stations for a particular cluster
        dist_x.append(len(values_x))
    distances_x.append(dist_x)
    #sys.exit()
    #dist = [[i for i in u if u > start_day] for u in time_idexs if u 
#plt.show()
    plt.xlabel("Position in time in days")
    plt.ylabel("Label")
    #%%-------------------------------------plottting cluster frequency densities for time -----------------------------------------------
    plt.figure(3)
    for i in range(len(distances_y)):
        plt.plot(steps, distances_y[i], color = colours[i], label = f"Label {i}")
    plt.legend(draggable = True)
    plt.title(f"Cluster frequencies over time | Step gap {round((step/(end_day-start_day)),2)} days")
    if more_stat== False: 
        return ""
    
    global station_positions, sti
    
    plt.clf()
    #CALCULATE UTC DATE TIME STEPS 
    
    alt_month_axes = int(alt_month/12)
    factors = {"DayNight": 0, "Mag>3": 1, "Mag>4": 2, "AvWindSpeed": 3}
    global Pearson_matrix
    Pearson_matrix = np.empty((len(station_list), n_clusters,2+len(weather.columns)))
    global current_station
    #print(Pearson_matrix.shape); sys.exit()
    
    #%% LOOPING OVER EACH STATION
    
    """
    
    IN THESE CODES INFORMATION ABOUT EACH STATION WILL BE DISPLAYED 
    INCLUDING STACK PLOTS AND CROSS CORRELATIONS
    
    
    """
    
    
    global all_labels_master; all_labels_master = []
    for u in range(0,len(station_list)):
        current_station = u #to avoid confusion with other variables 
        station_positions  = clustered_data.loc[clustered_data['Station'] == u, ['Time', "Label"]]
        station_positions["Time"] *= day_diff
        #PLOT HISTOGRAMMMMM
        #sys.exit(0)
        if produce_stack == True:
            fig, axes = plt.subplots(4,1,figsize = (10,15))
        #plt.figure(figsize =(10, 20))
        #plt.rcParams["figure.dpi"] = 
        
            for i in range(n_clusters):
                sti = station_positions.loc[station_positions['Label'] == i, 'Time']
                
                axes[0].plot(np.sort(sti), np.linspace(0, 1, len(sti), endpoint=False), label=f'Label {i}', color=colours[i])
        else: 
            for i in range(n_clusters):
                sti = station_positions.loc[station_positions['Label'] == i, 'Time']
        if produce_stack == True:
            ax4 = axes[1].twinx()
            ax5 = axes[2].twinx()
            lw = 2
            
            ax4.plot(steps_sec_old, mag4_sums, label = "Mag 4", linestyle="--", lw = lw, color = "white",  path_effects=[pe.Stroke(linewidth=3, foreground='y'), pe.Normal()])
            ax5.plot(steps_sec_old, mag4_sums, label = "Mag 4", linestyle="--",lw = lw, color = "white", path_effects=[pe.Stroke(linewidth=3, foreground='y'), pe.Normal()])
            ax4.plot(steps_sec, mag5_sums, label = "Mag 5", linestyle="--", lw=lw, color = "black", path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
            ax5.plot(steps_sec, mag5_sums, label = "Mag 5", linestyle="--", lw=lw, color = "black", path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
            ax4.plot(steps_sec, mag3_sums, label = "Mag 3", linestyle="--", lw=lw, color = "purple", path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
            ax5.plot(steps_sec, mag3_sums, label = "Mag 3", linestyle="--", lw=lw, color = "purple", path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
    
        global values_cul; values_cul = []; global array
        global sorted_times;sorted_times = station_positions.sort_values(by="Time", ascending = True)
        

        W = 100 #rolling constant (how many samples per window)
       
        #global all_labels;
        global all_labels; all_labels = []
        
        #sorted_times.insert(4, "Lab0Counts")
        sorted_times["Lab0Counts"] = 0
        #print(sorted_times.head()); sys.exit()
        for ii in range(num_clusters):
            lab = ii 
            sorted_times["Lab0Counts"] = sorted_times["Label"].rolling(W,min_periods =1).apply(cnt)
            sorted_times2 = sorted_times.groupby("Time", as_index = False)["Lab0Counts"].sum()
            data = sorted_times2.to_numpy()
            smo = savgol_filter(data[:,1],100,1,mode="nearest")
            
            time = data[:,0]
            """ADJUST AS NEEDED"""
            #time *= (420/duration) #adjust the axis for time 
            all_labels.append([time,smo])
            
        
    
    
    #fig,ax=plt.subplots()
        if produce_stack == True:
            for i in range(1,3):
                axes[i].stackplot(time,all_labels[0][1],all_labels[1][1], all_labels[2][1], all_labels[3][1], all_labels[4][1], all_labels[5][1])# all_labels[6][1], all_labels[7][1])
            
                axes[i].get_yaxis().set_ticks([0, W])
                #axes[i].get_xaxis().set_ticks([i for i in range(int(start_day*day_diff), int(end_day*day_diff), 31)],[str(i) for i in np.linspace(start_day, end_day, 31)])
                axes[i].set_ylabel("Density")
                #axes[i].legend(labels,loc = 'upper left')
                axes[i].set_xlabel('TIME IN DAYS')
                prop_cycle = plt.rcParams["axes.prop_cycle"]
                
                colors = colours
                axes[i].set_prop_cycle(color=colours)
            
           
            
            #axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
            #axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S UTC'))
            axes[1].legend(bbox_to_anchor=(1.2, 1))
            
            ax4.legend(bbox_to_anchor=(1.2, 1))
            axes[0].set_title(f"Histogram for station {u}", fontweight = "bold", fontsize = "large")
            #axes[1].set_yscale("log")
            axes[1].set_title(f"Propogation for station {u}")
            axes[0].set_ylabel("Cumulative distribution")
            axes[1].set_ylabel("Logirithimic frequency")
            axes[2].set_ylabel("Linear frequency of clusters")
            axes[2].set_xlabel("TIME IN DAYS")
            axes[3].set_xlabel("TIME IN DAYS")
            plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/%s%s{path_add_cluster}_Stack.png" %("TESTING" if testing == 1 else "","FLIP" if switch == True else ""))
            
            plt.close() #stops plotting in main terminal to save memoryh 
            
        
        #%% PLOT INVIDIUAL OCCURANCES OF EACH CLUSTER AT EACH STATION
        
        f, axes2 = plt.subplots(nrows=n_clusters, ncols=1)# sharey=True)
        pwm = day_night(all_labels, time)
        mag_compare = mag3_sums_smoothe; mag_num = 3 #CHANGE AT SAME TIME
        f.suptitle(f"Station {u}")
        
        for i,r in enumerate(labels):
            ax33 = axes2.flat[i]
            axop33 = ax33.twinx()
            ax33.plot(time,all_labels[i][1], label = str(i), color = colours[i])
            #print(np.max(all_labels[0][1]))
            pwm = [(r+1)*0.5*np.max(all_labels[i][1]) for r in pwm] #shifts it above 0 np.max(all_labels[i][1])
            #ax33.plot(time_total, pwm, "--", label = "Days", color = "lightsteelblue")
            global corr; corr = correlate(all_labels[i][1], pwm, mode = "same") 
            #global area_occ; area_occ = np.trapz(all_labels[i][1], time_total) #area underneath label
            #corr /= area_occ #normalise by area
            #print(np.max(all_labels[0][1]))
            #sys.exit()
            ax33.get_yaxis().set_ticks([0, np.max(all_labels[i][1])])
            #print(np.max(all_labels[0][1]))
            ####CHANGE THESE
            
            steps_sec = [i*day_diff for i in np.flip(steps_smooth)] #smoothing function is bigger
            
            
            #
            ##calculate pearson correlation coefficient 
            icorr = 0
            dummy1, dummy2, pearson = Pearson_correlation(all_labels[i][1],np.array(pwm))
            Pearson_matrix[u,i,0] = pearson
            *_,pearson = Pearson_correlation(all_labels[i][1],np.array(mag_compare))
            Pearson_matrix[u,i,1] = pearson
                
            #print(np.max(all_labels[0][1]))
            for index, (column_name, column_data) in enumerate(weather.items()):
                
                *_, pearson = Pearson_correlation(all_labels[i][1], np.array(column_data))
                Pearson_matrix[u,i,index+2] = pearson
            #p#rint(np.max(all_labels[0][1]))
            #%%
            # global compare_ratio; compare_ratio = np.max(all_labels[i][1])/np.max(compare)
            # global compare_normal; compare_normal = [i*compare_ratio  for i in compare]
            global compare_x_vals; compare_x_vals = np.linspace(0,max(steps_sec), len(time_total))
            # global compare_normal_interp; compare_normal_interp = np.interp(compare_x_vals, np.flip(steps_sec), compare_normal)
            #INTERPOLATION TO ENSURE MAG_COMPARE NORMAL IS THE SAME SIZE AS THE LABELS
            #global corr_eq; corr_eq = correlate(all_labels[i][1], compare_normal_interp, mode = "same")
            global corr_eq; corr_eq = correlate(X, Y, mode = "same")
            #corr_eq /= area_occ #normalisation by label area
            #ax33.plot(compare_x_vals, Y, "--")
            #ax33.plot(time, X, label = str(i), color = colours[i])
            #axop33.plot(time_total, corr_eq, "-", color = "dimgrey", label = "Mag>3 Correlation")
            #print(all_labels[i][1].shape); sys.exit()
            #global corr_lag; corr_lag = correlation_lags(all_labels[i][1].size, compare_normal_interp.size, mode = "same")
            #lag = corr_lag[np.argmax(corr_eq)]
            #axtext = axes2[i]
            #corrs[alt_month_axes, i,current_station] = np.max(corr_eq) #maxmimum correlation value for cpmpariosn 
            
            # axtext.text(1E6, 10, f'Max{np.max(corr_eq)}\nLag{lag}',
            #         horizontalalignment='left', fontsize = "large",
            #         verticalalignment='top', transform=ax.transAxes,
            #         bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            #axop33.set_ylabel("Absolute correlation")
            #PEARSON CROSS CORRELATION COEFFICIENT 
        #plt.close()
            
        #print(np.max(all_labels[0][1])); sys.exit()
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/{compare_name}%s%s%s{path_add_cluster}_Occurance.png" %("TESTING" if testing == 1 else "", "FLIP" if switch == True else "", {endtt.year, endtt.month}if alt_month == True else ""))
        plt.legend(loc = 'upper left')
        ax33.set_xlabel("Datapoints")
        fig.text(0.06, 0.5, 'Occurance', ha='center', va='center', rotation='vertical')
        fig.text(0.5, 0.5, 'Absolute correlation', ha='center', va='center', rotation='vertical')
        #print(np.max(all_labels[0][1])); sys.exit()
        all_labels_master.append(all_labels) #order: station, label, [time, occurance]
        #sys.exit()
        plt.close() 
        steps_sec = steps_sec_old #revert time steps back or original
    #%% GEOMETRIC ANALYSIS 
    sample_plot_rate = 100
    analysis = gma(all_labels_master,  "Distance_Epicentre", files[filenum][:-3], n_clusters,  len(station_list), colors = colours)
    global epi_dist; epi_dist = analysis.extract() 
    #analysis.plot_clusters_distance_time(epi_dist, sample_plot_rate)
    #analysis.trip_color(epi_dist, X_train_station, X_train_pos, labels_train, shading = "flat")
    analysis.stackplot(epi_dist,1000, omits = [1,4], lat_dim = latent_dim) #in units of 80 seconds
    #%%
    coloumns = [str(i) for i in station_list]
    #print(corrs)
    
    #%% CORRELATION HEATMAP
    
    if individual_corr_plots: 
        ration = 2
        fig, axes = plt.subplots(nrows = 2, 
                                 ncols = 5)
        
        print(column_names)
        for iii, rrr in enumerate(Pearson_matrix):
            ax2 = axes.flat[iii]
            im = ax2.imshow(Pearson_matrix[iii,:,:], cmap = "seismic")
            ax2.set_xticks(np.arange(len(column_names)), labels = column_names, fontsize = "xx-small")
            for i in range(Pearson_matrix.shape[1]):
                for j in range(Pearson_matrix.shape[2]):
                    corrnum = str(Pearson_matrix[iii,i, j])
                    # text = ax2.text(j, i, corrnum[1:4],
                    #        ha="center", va="center", color="w", fontsize = "xx-small")
            plt.setp(ax2.get_xticklabels(), rotation=90, ha="right",
                     rotation_mode="anchor")
            ax2.axes.yaxis.set_ticks([])
            ax2.set_title(f"{iii}")
        #fig.ylabel("Cluster number")
        fig.suptitle(f"Pearson Product Moment Coefficient testing = {testing}")
        cbar = fig.colorbar(im, ax = axes.ravel().tolist())
        cbar.set_label("Pearson coefficient")
        plt.savefig(f"/home/vsbh19/plots/summary_tables/%s%s%s{path_add_cluster}_Summary.png" %("TESTING" if testing == 1 else "", "FLIP" if switch == True else "", {endtt.year, endtt.month}if alt_month == True else ""))
    #%% PLOTTING GENERAL SUMMARY 
    global summary_pearson_av; summary_pearson_av = np.empty((Pearson_matrix.shape[1], Pearson_matrix.shape[2]))
    global summary_pearson_std; summary_pearson_std = np.copy(summary_pearson_av)
    for n in range(Pearson_matrix.shape[1]):
        for m in range(Pearson_matrix.shape[2]):
            summary_pearson_av[n,m] = np.average(Pearson_matrix[:,n,m])
            summary_pearson_std[n,m] = np.std(Pearson_matrix[:,n,m])
    fig, ax = plt.subplots(2,1, figsize =(5,9))
    #im = ax.imshow(summary_pearson)
    for i in np.arange(summary_pearson_av.shape[1]):
        ys = summary_pearson_av[:,i]
        stds = summary_pearson_std[:,i]
        #sizes = stds / (np.max(stds) * np.abs(ys))
        ax[0].scatter(np.ones((summary_pearson_av.shape[0])) * i,ys, s = stds*1000, c= colours[:n_clusters], marker = "o", facecolors = 'none')
        ax[1].scatter(np.ones((summary_pearson_av.shape[0])) *i, np.abs(ys)/stds, c = colors[:n_clusters], marker = "o", s = np.abs(ys)*1000)
    fig.suptitle(f"SummaryCorrelationfigureTesting={testing}")
    ax[0].axhline(0.29, 0, 5, linestyle = "--")
    ax[0].axhline(-0.29, 0, 5, linestyle = "--") #siginifcance for two tailed test?
    ax[1].axhline(1, 0, 5, linestyle = "--") #siginifcance for two tailed test?
    #ax[1].axhline(-1, 0, 5, linestyle = "--") #siginifcance for two tailed test?
    #ax[0].set_xlabel("Cluster Number")
    ax[0].set_ylabel(r"$\rho$")
    ax[1].set_ylabel(r"$\left|{\frac{\rho}{\sigma}} \right|$") 
    ax[1].set_title("Ratio coefficients to standard deviation")
    ax[1].set_xticks(np.arange(len(column_names)), column_names)
    ax[0].axes.xaxis.set_ticks([])
    plt.setp(ax[1].get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    plt.savefig(f"/home/vsbh19/plots/summary_tables/%s%s%s{path_add_cluster}_SummaryAllStations.png" %("TESTING" if testing == 1 else "", "FLIP" if switch == True else "", {endtt.year, endtt.month}if alt_month == True else ""))
    
    fig.tight_layout()
#-----------------------------------------------------------------------------------------------------------------------------------------
    #ti_scale = f.get("Time scale") [:]
#%%
secs = ['0', '8', '16', '24', '32']
se_basic = np.arange(0,40,8)
y_limits = [0,140] #y limits for presentation at 18, eliminating strong freq band in freq bins 0 to 15
y_limit_difference = y_limits[1] - y_limits[0]
fre_pos = np.linspace(0,140-y_limits[0], 4, dtype=int)
secs = [str(i*(duration/40)) for i in se_basic]
#fre = [str((i*(40/duration)))[0:2] for i in fre_pos]
try:
    fre = [str(i)[0:4] for i in np.linspace(0,max(fre_scale),4)]
except:
    fre = [str(i)[0:4] for i in np.linspace(0,8,4)]

#fre = ["0", "4", "8", "12","16", "20"]
#fre_pos = np.arange(0,24,4)
#fre = ["0", "4", "8", "12","16", "20"]
#fre_pos = np.arange(0,24,4)
secs_pos = np.arange(0,50,10)

from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
#plt.pcolormesh(X_train[20000,:,:,0])#; sys.exit()

#%%----------------------------------cluster size labels and clusters-----------------------

months = [0]#12,24]
len_month = len(months)
global corrs; corrs = np.ones((len_month, n_clusters, len(station_list)), dtype = float)
for alt_month in months:
    #alt_month = True #do we want to compare time series from different months with the cross correlations ? 
    factor = 12 if alt_month == True else 0 #if comparing different year
    endtt = file_times[filenum] - factor*month_diff #-MONTH DIFF 5 months before if comparing months 
    startt = endtt - month_diff
    if synthetic == False: 
        events = heythem("IU",["MAJO"], startt=endtt - month_diff,  endt=endtt, magnitude = minmag, verbose = 2)#; sys.exit()
    if testing == 0: 
        print_cluster_size(labels_train)
        print_all_clusters(X_train, labels_train,
                        n_clusters, X_train_pos, 
                        X_train_station,
                        alt_month = alt_month, reconst = train_reconst, plot_clusters_only=False)
                        #truths=truths_train)#; sys.exit()
    if testing == 1: 
        print_cluster_size(labels_test)
        print_all_clusters(X_test, labels_test,
                        n_clusters, X_test_pos, 
                        X_test_station,
                        alt_month = alt_month)
                        #truths=truths_train)#; sys.exit()
        
f, ax = plt.subplots(len(corrs),1)
#%%-----------------------------------------------------------------------------------------

lower_bound = 268170000
upper_bound = 268170000 + (5 * 3600*100)

last_loc = np.where(np.logical_and(lower_bound <= X_train_pos, X_train_pos <= upper_bound))
last_loc = np.reshape(last_loc, (len(last_loc[0]),))
last_loc = np.where(labels_train == 1)[0]
plot_var = X_train

#for u,idx in enumerate(np.random.randint(0,len(plot_var),25)):
#for u,idx in enumerate(last_loc[::200]):
for u,idx in enumerate(np.random.randint(0, len(X_train),25)):
    
    fig = plt.figure(figsize=(8,17))
    gs = GridSpec(nrows=3, ncols=3, width_ratios=[0.1,0.1,0.1], height_ratios = [1,0,1])
    fig.suptitle(f"#{u}, Reconstructed spectrogram number {idx} for {stationname}, time {str(X_train_pos[idx]/(3600*24*100))[0:3]}, station {X_train_station[idx]}")
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    eps = 0.5
    #Original spectrogram
    cb0 = ax0.imshow(np.reshape(X_train[idx,y_limits[0]:136, :40, :], (136-y_limits[0],40)))# norm = colors.LogNorm(vmin = 1e0, vmax = 1e1))
    #c#b0 = ax0.imshow(np.log1p(np.maximum(eps, np.reshape(X_val[idx, :140, :204, :], (140, 204))))
    ax0.set_ylabel('Frequency (Hz)')
    #ax0.set_xticks()
    ax0.set_xticks(secs_pos, secs)
    ax0.set_yticks(fre_pos, fre)
    ax0.set_xlabel('Time(s)')
    ax0.set_ylim([140,0])
    ax0.set_aspect(1)
    ax0.invert_yaxis()      
    #plt.colorbar(cb0, ax = ax0)
    if supervised == True:
        ax0.set_title(f'Original Spectrogram \nCLASS{truths_train[idx]}')
    else: 
        ax0.set_title('Original Spectrogram')
    
    #Latent space image representation
    cb1 = ax1.imshow(train_enc[idx].reshape(latent_dim,1), cmap='viridis')
    ax1.invert_yaxis()
    ax1.set_aspect(2)
    #plt.colorbar(cb1, ax = ax1)
    ax1.set_title('Latent Space')
    
    #Reconstructed Image
    cb2 = ax2.imshow(np.reshape(train_reconst[idx,y_limits[0]:, :, :] + eps, (136-y_limits[0],40))) #norm = colors.LogNorm(vmin = 1e0, vmax = 1e1))
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xticks(secs_pos, secs)
    ax2.set_xlabel('Time(s)')
    ax2.set_ylim([140,0])
    ax2.set_yticks(fre_pos, fre)
    
    ax2.set_aspect(1)
    ax2.invert_yaxis()      
    #plt.colorbar(cb2, ax = ax2)
    if supervised == True:
        ax2.set_title(f'Reconstructed  Spectrogram \n CLASS{truths_train[idx]}')
    else: 
        ax2.set_title('Reconstructed  Spectrogram')
    fig.tight_layout()
    fig.savefig(f'/home/vsbh19/plots/20 second snippets/%s%sDEC_original_embedded_reconst_{path_add_cluster}_{idx}.png' %("TESTING" if testing == 1 else "","FLIP" if switch == True else ""))
    plt.show()

sys.exit(1) #Completion stamp to shell 
