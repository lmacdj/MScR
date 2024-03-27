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
import matplotlib.pyplot as plt
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
from scipy.signal import savgol_filter, correlate, square
from drumplot import drumplot

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
except:
    filenum =7
    component = [2]
    stationname = "Sinosoid"
    duration = 240
    start_day =0
    end_day = 31.25
    norm = "l2"
    n_clusters = 4
    switch = "False"
drum_query = False
sample_rate = 100
samples_per_day = 60*60*24*sample_rate #seconds*mins*hours*sample rate in hertz
month_diff = 2700000 # seconds month
day_diff = 3600*24 #seconds in day
if files[filenum] == "Sinosoid.h5":
    synthetic = True #we dont want to dry and drag out events for synthetic data!
else:
    synthetic = False
minmag = 4
nobackupname = os.path.dirname("/nobackup/vsbh19/snovermodels/")
with h5py.File(nobackupname + f'/%sDEC_Training_LatentSpaceData_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.h5' %("FLIP" if switch == "True" else ""), 'r') as nf:
    print(nf.keys())
    #val_reconst = nf.get("Val_ReconstructData")[:]
    #val_enc = nf.get("Val_EncodedData")[:]
    #enc_train =nf.get("Train_EncodedData")[:]
    train_reconst = nf.get("Train_ReconstructData")[:]
    labels_train = nf.get("Labels_Train")[:]
    #labels_test = nf.get("Labels_Test")[:]
    #train_enc = nf.get("Train_Reconst")[:]
    #train_reconst = nf.get("Train_ReconstructData")[:]
# with h5py.File(nobackupname + f'/Training_LatentSpaceData_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5', 'w') as fu:
#     print(fu.keys())
#     train_enc = fu.get("TrainEnc")[:]

with h5py.File(f"/nobackup/vsbh19/training_datasets/%sX_train_X_val_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.h5" %("FLIP" if switch == "True" else ""), "r") as f:
    #print(f.keys()); sys.exit()
    X_val = f.get("X_val")[:]
    X_train = f.get("X_train")[:]
    fre_scale = f.get("Frequency scale")[:]
    train_enc = f.get("Trained_Encoded_Data")[:]
    X_train_pos = f.get("X_train_pos")[:]
    X_val_pos = f.get("X_val_pos")[:]
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

if drum_query == True: 
    directory = os.path.join("/nobackup/vsbh19/h5_files/")
    with h5py.File(directory + files[filenum]) as f:
        stations = [i for i in f.keys()]
        print(stations)
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
    fig1=plt.figure() 
    global time_idxes, station_idxes, fre
    global clustered_data
    clustered_data = pd.DataFrame(columns=["Label", "Spectrogram", "Time", "Station"])
    amounts = []
    
    for cluster_num in range(0,num_clusters):
        fig = plt.figure(figsize=(14,5))
        num_labels = max(labels) + 1
        cnt = 0
        num_fig = 8
        global  time_idx, station_idx, spec_num
        label_idx = np.where(labels==cluster_num)[0] #where in the labels located?
        cluster_array = [cluster_num for i in label_idx]
        time_idx = [positions[i] for i in label_idx] #where are the times for this cluster?
        station_idx = [stations[i] for i in label_idx] #where are the stations for this cluster?
        spec_num = [i for i in range(0, len(label_idx))]
        amount = len(label_idx)
        amounts.append(amount)
        #time_idxes.append(time_idx)
        #station_idxes.append(station_idx)
        global to_dataframe
        to_dataframe = pd.DataFrame({
                        "Label": cluster_array,
                        "Spectrogram": spec_num, 
                        "Time": [i/samples_per_day + start_day for i in time_idx], #this converts time samples to days
                        "Station": station_idx})
        clustered_data = pd.concat([clustered_data, to_dataframe], ignore_index=True)
        clustered_data.to_csv("out.csv")
        gs = GridSpec(1,num_fig,width_ratios = [1 for i in range(num_fig)])
        if len(label_idx) < num_fig:
            for i in range(len(label_idx)):
                ax = plt.subplot(gs[i])
                ax.imshow(np.reshape(data[label_idx[i],:], (140,41)))
                ax.set_ylabel('Frequency (Hz)')
                ax.set_ylim([0,140])
                ax.set_xticks(secs_pos, secs)
                ax.set_xlabel('Time(s)')
                ax.set_yticks(fre_pos, fre)
                ax.set_aspect(1)
                #ax.invert_yaxis()
                #plt.colorbar()
        else: 
            for i in range(0,num_fig):
                cnt = cnt + 1
                ax = plt.subplot(gs[i])
                ax.imshow(np.reshape(data[label_idx[random.randint(0,len(label_idx))],:], (140,41)))
                ax.set_ylabel('Frequency (Hz)')
                ax.set_ylim([0,140])
                ax.set_xlabel('Time(s)')
                ax.set_yticks(fre_pos, fre)
                ax.set_xticks(secs_pos, secs)
                ax.set_aspect(1)
                #ax.invert_yaxis()
                #plt.colorbar()
             
            plt.savefig(f"/home/vsbh19/plots/Clusters/%s{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_Cluster{cluster_num}.png" %("FLIP" if switch == True else ""))
        plt.suptitle(f"Label {cluster_num} \t #:{len(label_idx)} \t %:{int(100*len(label_idx)/len(data))}", ha='left', va='center', fontsize=28)    
        plt.tight_layout()
    
    ##----DRUM PLOT ----------------------------
    
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
        plots = drumplot(drum_array,clustered_data, duration, n_clusters, 
                     start_day_drum, end_day_drum, station_num, 
                     end_of_series_time = file_times[filenum], 
                     clipping = True, colors = colours, amount = 10, nrows=4); 
        plots.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/%sSERIESCLUSTERS_{station_num}_{component}_{duration}_{norm}_Cluster{n_clusters}_Examples.png" %("FLIP" if switch == True else ""))
        plt.clf()
        sys.exit("DRUMPLOTS CREATED")
        
     
    
    ###-------PLOT LABELS IN TIME AND SPACE-----------------------
    plt.show()
    #sys.exit()
    plt.figure(2)
    
    global step
    step = (end_day-start_day)/50 #in days !!
    sliding_step = step/16
    global steps, values_y, r, reader, steps_sec, event_mag, event_times, reader, mag5
    steps = np.linspace(start_day, end_day, num = int((end_day-start_day)/step)) #UNIT: DAYS
    #steps = np.linspace(start_day, end_day, num = int((end_day-start_day) - step /sliding_step)) #UNIT: DAYS
    if supervised == True:  
        try: 
            truths 
        except: 
            sys.exit("YOU NEED TO DEFINE WHICH TRUTHS YOURE USING AS A KWARG truths =")
        cm = confusion_matrix(truths, labels)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/Sinosoid.h5/Confusion_C{n_clusters}_L{max(truths)}.png")
        # print("ACCRUACY", accuracy_score(truths, labels))
        # print("PRECISION", precision_score(truths, labels))
        
        
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
    #print(event_times)
    
    steps_sec = [i*day_diff  for i in np.flip(steps)] #steps converted into days
    #STEPS IS FLIPPED FOR A GIVEN ORDER AS TIME OF EVENTS INCREASES AWAY FROM EARTHQUAKE
    #sys.exit()
    global sort, sort_fil, dist_y, dist_x, distances_y, distances_x
    distances_y , distances_x = [],[]
    global station_list
    station_list = np.arange(np.min(stations), np.max(stations), dtype = int)
    fig, ax = plt.subplots()
    for i in range(num_clusters):
        #length = len(time_idxes[i])
        length = amounts[i] 
        label_plot = int(i)*np.ones(shape = length) #linspace for label
        time_idx = clustered_data.loc[clustered_data['Label'] == i, 'Time']
        ax.scatter(time_idx, label_plot, color = colours[i], s = 0.001)
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
    #-------------------------------------plottting cluster frequency densities for time -----------------------------------------------
    plt.figure(3)
    for i in range(len(distances_y)):
        plt.plot(steps, distances_y[i], color = colours[i], label = f"Label {i}")
    plt.legend(draggable = True)
    plt.title(f"Cluster frequencies over time | Step gap {round((step/(end_day-start_day)),2)} days")
    if more_stat== False: 
        return ""
    global mag4, mag4list, mag4_sums, mag5_sums
    
    #---------------------------------------plotting cluster frequencies for station-----------------------------------------------
    mag4 = reader.loc[reader["Magnitude"]<=5, "Time"]
    
    mag5 = reader.loc[reader["Magnitude"]>=5, "Time"]
    mag4_sums = []; mag5_sums = []
    for u in steps_sec:
        mag4_contain, mag5_contain = [v for v in mag4 if u <= v <= u+step*day_diff], [v for v in mag5 if u <= v <= u+step*day_diff]
        mag4_sums.append(len(mag4_contain))
        mag5_sums.append(len(mag5_contain))
        
    global station_positions, sti
    
    plt.clf()
    #CALCULATE UTC DATE TIME STEPS 
    for u in range(0,len(station_list)+1):
        
        station_positions  = clustered_data.loc[clustered_data['Station'] == u, ['Time', "Label"]]
        station_positions["Time"] *= day_diff
        #PLOT HISTOGRAMMMMM
        #sys.exit(0)
        fig, axes = plt.subplots(4,1,figsize = (10,15))
        #plt.figure(figsize =(10, 20))
        #plt.rcParams["figure.dpi"] = 
        for i in range(n_clusters):
            sti = station_positions.loc[station_positions['Label'] == i, 'Time']
            
            axes[0].plot(np.sort(sti), np.linspace(0, 1, len(sti), endpoint=False), label=f'Label {i}', color=colours[i])
        
        #for i in reader["Time"]:
            #axes[0].axvline(x=i, color='red', linestyle='--')
        
        ax4 = axes[1].twinx()
        ax5 = axes[2].twinx()
        lw = 2
        ax4.plot(steps_sec, mag4_sums, label = "Mag 4", linestyle="--", lw = lw, color = "white",  path_effects=[pe.Stroke(linewidth=3, foreground='y'), pe.Normal()])
        ax5.plot(steps_sec, mag4_sums, label = "Mag 4", linestyle="--",lw = lw, color = "white", path_effects=[pe.Stroke(linewidth=3, foreground='y'), pe.Normal()])
        ax4.plot(steps_sec, mag5_sums, label = "Mag 5", linestyle="--", lw=lw, color = "black", path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
        ax5.plot(steps_sec, mag5_sums, label = "Mag 5", linestyle="--", lw=lw, color = "black", path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
        

    
        global values_cul; values_cul = []; global array
        global sorted_times;sorted_times = station_positions.sort_values(by="Time", ascending = True)
        
        #plt.figure(6)
        #plt.plot(sorted_times); sys.exit()
        def cnt(x): 
            prev_count = 0
            for i in x: 
                if i == lab:
                    prev_count += 1 
            return prev_count
        W = 25 #rolling constant (how many samples per window)
       
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
            global time
            time = data[:,0]
            """ADJUST AS NEEDED"""
            #time *= (420/duration) #adjust the axis for time 
            all_labels.append([time,smo])
            
        labels = np.arange(0,num_clusters)
        print(labels)
        
        #fig,ax=plt.subplots()
        # all_labels_transposed = np.array(all_labels).T.tolist()
        # global times_2d; times_2d = np.array([time]).T
        # ax4.stackplot(times_2d[0], *all_labels_transposed)
        for i in range(1,3):
            axes[i].stackplot(time,all_labels[0][1],all_labels[1][1], all_labels[2][1], all_labels[3][1])# all_labels[4][1], all_labels[5][1], all_labels[6][1], all_labels[7][1])
        
            axes[i].get_yaxis().set_ticks([0, W])
            #axes[i].get_xaxis().set_ticks([i for i in range(int(start_day*day_diff), int(end_day*day_diff), 31)],[str(i) for i in np.linspace(start_day, end_day, 31)])
            axes[i].set_ylabel("Density")
            axes[i].legend(labels,loc = 'upper left')
            axes[i].set_xlabel('TIME IN DAYS')
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            global colors
            colors = colours
            axes[i].set_prop_cycle(color=colours)
        
        current = sorted_times["Label"].iloc[0] #CURRENT LABEL
        counter_old = 1 #CURRENT TIME 
        alpha = 0.05
        for q in range(len(sorted_times)): #iterate throigh all the times
           
           if sorted_times["Label"].iloc[q] != current: # IF LABEL DIFFERS
               #REQUIREMENT TO MOVE ON LABEL HAS BEEN ACHIEVED SO AN XVSPAN WILL BE PLOTTED
                axes[3].axvspan(counter_old*duration+(duration/4), 
                            q*duration+(duration/4), alpha = alpha, color = colours[current], ymin = 0.1,   label = f"Label {current}")
                # axes[2].axvspan(counter_old*240+80, 
                #             q*240+80, alpha = alpha, color = colours[current],  ymin = 0.1,  label = f"Label {current}")
                counter_old = q 
                current = sorted_times["Label"].iloc[q] 
        #axes[3].legend(bbox_to_anchor=(1.2, 1))
        
        #axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        #axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S UTC'))
        axes[1].legend(bbox_to_anchor=(1.2, 1))
        
        ax4.legend(bbox_to_anchor=(1.2, 1))
        axes[0].set_title(f"Histogram for station {u}", fontweight = "bold", fontsize = "large")
        axes[1].set_yscale("log")
        axes[1].set_title(f"Propogation for station {u}")
        axes[0].set_ylabel("Cumulative distribution")
        axes[1].set_ylabel("Logirithimic frequency")
        axes[2].set_ylabel("Linear frequency of clusters")
        axes[2].set_xlabel("TIME IN DAYS")
        axes[3].set_xlabel("TIME IN DAYS")
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/%s{files[filenum][:-3]}_{u}_{component}_{duration}_{norm}_Cluster{n_clusters}_Stack.png" %("FLIP" if switch == True else ""))
        
        plt.close() #stops plotting in main terminal to save memoryh 
        f, axes2 = plt.subplots(nrows=n_clusters, ncols=1)# sharey=True)
        for i,r in enumerate(labels):
            ax33 = axes2.flat[i]
            axop33 = ax33.twinx()
            ax33.plot(time,all_labels[i][1], label = str(i), color = colours[i])
            end_points = np.max(time)
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
            global time_total; time_total = np.linspace(0,np.max(time), len(all_labels[i][1])) #takes into consideration there will be gapes in the actual time
            
            fre_sq = 2*np.pi*(1/day_diff)
            #global phased_period; phased_period = [i+phase_shift for i in period*time_total]
            global pwm; pwm = square(fre_sq*time_total +phase_shift , duty = 0.5)
            pwm = [(r+1)*0.5*np.max(all_labels[i][1]) for r in pwm] #shifts it above 0 np.max(all_labels[i][1])
            #pwm.append(end)
            #pwm[]
            ax33.plot(time_total, pwm, "--", label = "Days", color = "lightsteelblue")
            global corr; corr = correlate(all_labels[i][1], pwm, mode = "same") 
            #corr *= np.max(all_labels[i][1])/np.max(corr) # normalises the correlation arrays
            axop33.plot(time_total, corr, "-", color = "black", label = "Correlation")
            #sys.exit()
            ax33.get_yaxis().set_ticks([0, np.max(all_labels[i][1])])
            
            #axop33.set_ylabel("Absolute correlation")
        
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/%s{files[filenum][:-3]}_{u}_{component}_{duration}_{norm}_Cluster{n_clusters}_Occurance.png" %("FLIP" if switch == True else ""))
        plt.legend(loc = 'upper left')
        ax33.set_xlabel("Datapoints")
        fig.text(0.06, 0.5, 'Occurance', ha='center', va='center', rotation='vertical')
        fig.text(0.5, 0.5, 'Absolute correlation', ha='center', va='center', rotation='vertical')
        
        #sys.exit()
        plt.close() 
        
        f, axes9 = plt.subplots(nrows = n_clusters, ncols = n_clusters)
        #for i in range(0, n_clusters**2):
        for i in range(n_clusters):
            for u in range(n_clusters):
                #axes9 = axes9[i,u].flat[i,u]
                if i==u:
                    axes9[i,u].scatter(all_labels[i][1], all_labels[u][1], color = colours[i], s = 0.001)
                else: 
                    axes9[i,u].scatter(all_labels[i][1], all_labels[u][1], s = 0.001)
        f.suptitle(f"Scatters for station {u}")
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/%s{files[filenum][:-3]}_{u}_{component}_{duration}_{norm}_Cluster{n_clusters}_CrossCor.png" %("FLIP" if switch == True else ""))
        plt.close()
        #distance = (all_labels[1][1]**2 + all_labels[2][1])**2
        #sys.exit()
#-----------------------------------------------------------------------------------------------------------------------------------------
    #ti_scale = f.get("Time scale") [:]

secs = ['0', '8', '16', '24', '32']
se_basic = np.arange(0,40,8)
fre_pos = np.array([0,46,93,140])
secs = [str(i*(duration/40)) for i in se_basic]
#fre = [str((i*(40/duration)))[0:2] for i in fre_pos]
fre = [str(i)[0:4] for i in np.linspace(0,max(fre_scale),4)]

#fre = ["0", "4", "8", "12","16", "20"]
#fre_pos = np.arange(0,24,4)
#fre = ["0", "4", "8", "12","16", "20"]
#fre_pos = np.arange(0,24,4)
secs_pos = np.arange(0,50,10)

from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors

#----------------------------------cluster size labels and clusters-----------------------

endtt = file_times[filenum]
startt = endtt - month_diff
if synthetic == False: 
    events = heythem("IU",["MAJO"], startt=endtt - month_diff,  endt=endtt, magnitude = minmag, verbose = 2)#; sys.exit()
print_cluster_size(labels_train)

print_all_clusters(X_train, labels_train,
                   n_clusters, X_train_pos, 
                   X_train_station, truths=truths_train)#; sys.exit()

#-----------------------------------------------------------------------------------------
for u,idx in enumerate(np.random.randint(0,len(X_train),25)):
    fig = plt.figure(figsize=(8,17))
    gs = GridSpec(nrows=3, ncols=3, width_ratios=[0.1,0.1,0.2], height_ratios = [1,0,1])
    fig.suptitle(f"#{u}, Reconstructed spectrogram number {idx} for {stationname}")
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    eps = 0.5
    #Original spectrogram
    cb0 = ax0.imshow(np.reshape(X_train[idx,:136, :40, :], (136,40)))# norm = colors.LogNorm(vmin = 1e0, vmax = 1e1))
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
    cb1 = ax1.imshow(train_enc[idx].reshape(14,1), cmap='viridis')
    ax1.invert_yaxis()
    ax1.set_aspect(2)
    #plt.colorbar(cb1, ax = ax1)
    ax1.set_title('Latent Space')
    
    #Reconstructed Image
    cb2 = ax2.imshow(np.reshape(train_reconst[idx,:, :, :] + eps, (136,40))) #norm = colors.LogNorm(vmin = 1e0, vmax = 1e1))
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
    fig.savefig(f'/home/vsbh19/plots/20 second snippets/%sDEC_original_embedded_reconst_{files[filenum][:-3]}_{stationname}_{duration}_C{n_clusters}_{idx}.png' %("FLIP" if switch == True else ""))
    plt.show()

sys.exit(1) #Completion stamp to shell 
