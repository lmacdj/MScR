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
except:
    filenum = 5
    component = [2]
    stationname = "all"
    duration = 240
    start_day =0
    end_day = 31.5
    norm = "l2"
    n_clusters = 6
    
sample_rate = 100
samples_per_day = 3600*24*sample_rate
month_diff = 2700000 # seconds month
day_diff = 3600*24 #seconds in day
if files[filenum] == "Sinosoid.h5":
    synthetic = True #we dont want to dry and drag out events for synthetic data!
else:
    synthetic = False
minmag = 4
nobackupname = os.path.dirname("/nobackup/vsbh19/snovermodels/")
with h5py.File(nobackupname + f'/DEC_Training_LatentSpaceData_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.h5', 'r') as nf:
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

with h5py.File(f"/nobackup/vsbh19/training_datasets/X_train_X_val_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.h5" , "r") as f:
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
                        "Time": [i/samples_per_day + start_day for i in time_idx],
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
             
            plt.savefig(f"/home/vsbh19/plots/Clusters/{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_Cluster{cluster_num}.png")
        plt.suptitle('Label {}'.format(cluster_num), ha='left', va='center', fontsize=28)    
        plt.tight_layout()
    
    ###-------PLOT LABELS IN TIME AND SPACE-----------------------
    plt.show()
    #sys.exit()
    plt.figure(2)
    colours = ["red", "blue", "green", "purple",  "orange", "black"]#; sys.exit()
    global step
    step = (end_day-start_day)/200 #in days !!
    sliding_step = step/16
    global steps, values_y, r, reader, steps_sec, event_mag, event_times, reader, mag5
    steps = np.linspace(start_day, end_day, num = int((end_day-start_day)/step)) #UNIT: DAYS
    steps = np.linspace(start_day, end_day, num = int((end_day-start_day) - step /sliding_step)) #UNIT: DAYS
    if supervised == True: 
        try: 
            truths 
        except: 
            sys.exit("YOU NEED TO DEFINE WHICH TRUTHS YOURE USING")
        cm = confusion_matrix(truths, labels)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        print("ACCRUACY", accuracy_score(truths, labels))
        print("PRECISION", precision_score(truths, labels))
        
        
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
    plt.show()
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
        ax4.plot(steps_sec, mag4_sums, label = "Mag 4", linestyle="--")
        ax5.plot(steps_sec, mag4_sums, label = "Mag 4", linestyle="--")
        ax4.plot(steps_sec, mag5_sums, label = "Mag 5", linestyle="--")
        ax5.plot(steps_sec, mag5_sums, label = "Mag 5", linestyle="--")
        
        """
        
        
        
        
        *****************************
        
        
        
        STEFAN  this is where i get issues 
        
        
        
        
        
        
        ********************
        
        
        
        
        """
        
        global values_cul; values_cul = []; global array
        global sorted_times;sorted_times = station_positions.sort_values(by="Time", ascending = True)
        #plt.figure(6)
        #plt.plot(sorted_times); sys.exit()
        for i in range(n_clusters):
            #ti = sorted_times.loc[sorted_times['Label'] == i, 'Time'].copy()
            #sti = station_positions.loc[station_positions['Label'] == i, 'Time']
            # plt.figure(6)
            # plt.plot(np.sort(sorted_times), marker = "o", ms = 0.01); sys.exit()
            #global values_y
            #PLOTTING OCCURANCES OF LBABLES IN TIME 
            values_y = []
            for z,q in enumerate(steps): #plots each occurance per 6 hours of day
            #for z in range(850):
                #print(q); sys.exit()
                start_time =  z* sliding_step * day_diff 
                end_time = ((z * sliding_step) + step)* day_diff  #was z originally 
                #array = [v for v in sti if start_time <= v <= end_time]
                array = sorted_times[(sorted_times['Time'] >= start_time) & (sorted_times['Time'] <= end_time) & (sorted_times['Label'] == i)]
                values_y.append(len(array))
            values_cul.append(values_y)
            # axes[1].plot(steps_sec, values_y, color = colours[i], label = f"Label {i}")
            # axes[2].plot(steps_sec, values_y, color = colours[i], label = f"Label {i}")
            plt.figure(7)
            plt.scatter(steps_sec, values_y, label=f"Label {i}", color=colours[i], marker='o')
            plt.plot(np.sort(sti))
        #stacks = np.vstack([q for q in values_cul])
        axes[1].stackplot(steps_sec, *values_cul, labels=[f"Label {i}" for i in range(n_clusters)], colors=colours, baseline = "zero")
        axes[2].stackplot(steps_sec, *values_cul, labels=[f"Label {i}" for i in range(n_clusters)], colors=colours, baseline = "zero")
    
        """
        end of issues
        
        
        
        
        
        
        
        
        
        """
        #sys.exit()
        
        
        current = sorted_times["Label"].iloc[0] #CURRENT LABEL
        counter_old = 1 #CURRENT TIME 
        alpha = 0.05
        for q in range(len(sorted_times)): #iterate throigh all the times
           
           if sorted_times["Label"].iloc[q] != current: # IF LABEL DIFFERS
               #REQUIREMENT TO MOVE ON LABEL HAS BEEN ACHIEVED SO AN XVSPAN WILL BE PLOTTED
                axes[3].axvspan(counter_old*240+80, 
                            q*240+80, alpha = alpha, color = colours[current], ymin = 0.1,   label = f"Label {current}")
                # axes[2].axvspan(counter_old*240+80, 
                #             q*240+80, alpha = alpha, color = colours[current],  ymin = 0.1,  label = f"Label {current}")
                counter_old = q 
                current = sorted_times["Label"].iloc[q] 
          
    
        #mag4.index = pd.to_datetime(mag4.index, utc = True)
        

# Create a rolling window of size 7 (adjust as needed)
        

# Display the result
        
        dist_y.append(len(values_y))#;print(len(values_y)); sys.exit()
    
        
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
        axes[2].set_xlabel("TIME IN SECONDS")
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/{files[filenum][:-3]}_{u}_{component}_{duration}_{norm}_Cluster{cluster_num}.png")
        #plt.tight_layout()
        #sys.exit()
#-----------------------------------------------------------------------------------------------------------------------------------------
    #ti_scale = f.get("Time scale") [:]

secs = ['0', '8', '16', '24', '32']
se_basic = np.arange(0,40,8)
fre_pos = np.arange(0,24,4)
secs = [str(i*(duration/40)) for i in se_basic]
fre = [str((i*(40/duration)))[0:2] for i in fre_pos]

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
print_all_clusters(X_train, labels_train, n_clusters, X_train_pos, X_train_station); sys.exit()
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
    ax0.set_ylim([80,0])
    ax0.set_aspect(1)
    ax0.invert_yaxis()      
    plt.colorbar(cb0, ax = ax0)
    if supervised == True:
        ax0.set_title(f'Original Spectrogram \nCLASS{truths_train[idx]}')
    else: 
        ax0.set_title('Original Spectrogram')
    
    #Latent space image representation
    cb1 = ax1.imshow(train_enc[idx].reshape(14,1), cmap='viridis')
    ax1.invert_yaxis()
    ax1.set_aspect(2)
    plt.colorbar(cb1, ax = ax1)
    ax1.set_title('Latent Space')
    
    #Reconstructed Image
    cb2 = ax2.imshow(np.reshape(train_reconst[idx,:, :, :] + eps, (136,40))) #norm = colors.LogNorm(vmin = 1e0, vmax = 1e1))
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xticks(secs_pos, secs)
    ax2.set_xlabel('Time(s)')
    ax2.set_ylim([80,0])
    ax2.set_yticks(fre_pos, fre)
    
    ax2.set_aspect(1)
    ax2.invert_yaxis()      
    plt.colorbar(cb2, ax = ax2)
    if supervised == True:
        ax2.set_title(f'Reconstructed  Spectrogram \n CLASS{truths_train[idx]}')
    else: 
        ax2.set_title('Reconstructed  Spectrogram')
    fig.tight_layout()
    fig.savefig(f'/home/vsbh19/plots/20 second snippets/DEC_original_embedded_reconst_{files[filenum][:-3]}_{stationname}_{duration}_C{n_clusters}_{idx}.png')
    plt.show()

sys.exit(1) #Completion stamp to shell 
