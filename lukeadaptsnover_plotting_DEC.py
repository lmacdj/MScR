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
import random
import numpy as np
import h5py
import sys
import os
import pandas as pd
from retrieve_earthquake_prelim import heythem
files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5","ev0000447288.h5", "ev0000734973.h5", "Sinosoid.h5"]
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
        
def print_all_clusters(data, labels, num_clusters, positions, stations):
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
    fig1=plt.figure() 
    global time_idxes, station_idxes, fre
    #time_idxes = []; station_idxes = []
    
    
    global clustered_data
    clustered_data = pd.DataFrame(columns=["Label", "Spectrogram", "Time", "Station"])
    amounts = []
    for cluster_num in range(0,num_clusters):
        fig = plt.figure(figsize=(14,5))
        num_labels = max(labels) + 1
        cnt = 0
        num_fig = 10
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
        gs = GridSpec(1,num_fig,width_ratios = [1 for i in range(num_fig)])
        if len(label_idx) < num_fig:
            for i in range(len(label_idx)):
                ax = plt.subplot(gs[i])
                ax.imshow(np.reshape(data[label_idx[i],:], (140,41)))
                ax.set_ylabel('Frequency (Hz)')
                ax.set_ylim([0,20])
                ax.set_xticks(secs_pos, secs)
                ax.set_xlabel('Time(s)')
                ax.set_yticks(fre_pos, fre)
                ax.set_aspect(10)
                #ax.invert_yaxis()
                #plt.colorbar()
        else: 
            for i in range(0,num_fig):
                cnt = cnt + 1
                ax = plt.subplot(gs[i])
                ax.imshow(np.reshape(data[label_idx[random.randint(0,len(label_idx))],:], (140,41)))
                ax.set_ylabel('Frequency (Hz)')
                ax.set_ylim([0,20])
                ax.set_xlabel('Time(s)')
                ax.set_yticks(fre_pos, fre)
                ax.set_xticks(secs_pos, secs)
                ax.set_aspect(10)
                #ax.invert_yaxis()
                #plt.colorbar()
             
            plt.savefig(f"/home/vsbh19/plots/Clusters/{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_Cluster{cluster_num}.png")
        plt.suptitle('Label {}'.format(cluster_num), ha='left', va='center', fontsize=28)    
        plt.tight_layout()
    #time_idxes = np.asarray(time_idxes, dtype = float)
    #station_idxes = np.asarray(station_idxes, dtype = float)
    
    
    ###-------PLOT LABELS IN TIME AND SPACE-----------------------
    plt.show()
    
    plt.figure(2)
    colours = ["red", "blue", "green", "purple",  "orange", "black"]#; sys.exit()
    
    step = (end_day-start_day)/15 #in days !!
    global steps, values_y
    steps = np.linspace(start_day, end_day, num = int((end_day-start_day)/step))
    global sort, sort_fil, dist_y, dist_x, distances_y, distances_x
    distances_y , distances_x = [],[]
    global station_list
    station_list = np.arange(np.min(stations), np.max(stations), dtype = int)
    for i in range(num_clusters):
        #length = len(time_idxes[i])
        length = amounts[i] 
        label_plot = int(i)*np.ones(shape = length) #linspace for label
        time_idx = clustered_data.loc[clustered_data['Label'] == i, 'Time']
        plt.scatter(time_idx, label_plot, color = colours[i], s = 0.001)
        dist_y = []
        dist_x =[]
        for z,u in enumerate(steps): #plots each occurance per 6 hours of day
            global values
            values_y = [v for v in time_idx if u <= v <= u+step]
            
            dist_y.append(len(values_y))#;print(len(values_y)); sys.exit()
            
            #i2 = [time_idxes.index(r) for r in time_idxes[i] if r <= steps[z] + step and r >= steps[z]]
            #dist = [time_idxes[i] for i in i2]
            #dist = [q for r,q in enumerate(time_idxes[i])]
            #dist.append(sort_fil)
        distances_y.append(dist_y)
        station_idx = clustered_data.loc[clustered_data['Label'] == i, 'Station']
        for c,q in enumerate(station_list):
            values_x = [v for v in station_idx if q-0.5 <= v <= q+0.5] #returns an array of all the stations for a particular cluster
            dist_x.append(len(values_x))
        distances_x.append(dist_x)
        #sys.exit()
        #dist = [[i for i in u if u > start_day] for u in time_idexs if u 
    
    plt.xlabel("Position in time in days")
    plt.ylabel("Label")
    #-------------------------------------plottting cluster frequency densities for time -----------------------------------------------
    plt.figure(3)
    legend = {}
    for i in range(len(distances_y)):
        plt.plot(steps, distances_y[i], color = colours[i], label = f"Label {i}")
    plt.legend(draggable = True)
    plt.title(f"Cluster frequencies over time | Step gap {round((step/(end_day-start_day)),2)} days")
    if more_stat== False: 
        return ""
    #---------------------------------------plotting cluster frequencies for station-----------------------------------------------
    # fig = plt.figure(4)
    # for i in range(len(time_idxes)):
    #     plt.scatter(time_idxes[i], station_idxes[i], color = colours[i], s= 0.1)
    # fig = plt.figure(5)
    # ax12 = fig.add_subplot(111, projection  = "3d")
    # for i in range(len(time_idxes)):
    #     #ax.plot(steps, distances_y[i], station_idxes[i], color = colours[i])
    #     ax12.scatter(time_idxes[i], station_idxes[i], i*np.ones(len(time_idxes[i])), s = 0.1, color = colours[i])
    # ax12.set_xlabel("Time"); ax12.set_ylabel("Station Num"); ax12.set_zlabel("Label")
    # ax12.legend()
    # ax12.view_init(elev=20, azim = 30)
    # plt.show()
    
    time_for_stats = []
    # for u in range(len(stations)-1): 
    #     for i in range(len(num_clusters)): #for each label 
    #         global station_positions, distances_stat
    #         #station_positions = [index for index, q in enumerate(station_idxes[i]) if q == u] #station positions for label
    #         #station_position_in_time = [time_idxes[i][q] for q in station_idxes[i] if q == u]
    #         #distances_stat = [v for v in  cluster if u <= v <= u+step]
    #         time_for_stat = [time_idxes[i][q] for q in station_positions]
    #         time_for_stats.append(time_for_stat)
    global station_positions
    plt.clf()
    for u in range(0,len(station_list)+1):
        print(u)
        station_positions  = clustered_data.loc[clustered_data['Station'] == u, ['Time', "Label"]]
        #PLOT HISTOGRAMMMMM
        #sys.exit(0)
        #plt.clf()
        plt.figure(4+ 2*u)
        for i in range(n_clusters):
            sti = station_positions.loc[station_positions['Label'] == i, 'Time']
            plt.plot(np.sort(sti), np.linspace(0, 1, len(sti), endpoint=False), label=f'Label {i}', color=colours[i])
        plt.figure(2*u+5)
        for i in range(n_clusters):
            sti = station_positions.loc[station_positions['Label'] == i, 'Time']
            #global values_y
            values_y = []
            for z,q in enumerate(steps): #plots each occurance per 6 hours of day
                values_y.append(len([v for v in sti if q <= v <= q+step]))
            plt.plot(steps, values_y, color = colours[i], label = f"Label {i}")
        plt.legend()
        plt.title(f"Propogation for station {u}")
        plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/{files[filenum][:-3]}_{u}_{component}_{duration}_{norm}_Cluster{cluster_num}.png")
        #sys.exit()
    
    
    
    
    # for z,u in enumerate(steps): #plots each occurance per 6 hours of day
    #     global values
    #     values_y = [v for v in time_for_stat if u <= v <= u+step]
    # distances = [v for v in time_idxes[i] if u <= v <= u+step]
    # for u in range(len(stations)):
    #     plt.plot(steps, distances_stat[i])
        
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
events = heythem("IU",["MAJO"], startt="2014-07-04T19:22:00.0",  endt="2014-07-11T19:22:00.0", magnitude = 5.0, verbose = 1)
print_cluster_size(labels_train)
print_all_clusters(X_train, labels_train, n_clusters, X_train_pos, X_train_station)#; sys.exit()
#-----------------------------------------------------------------------------------------
for u,idx in enumerate(np.random.randint(0,len(X_val),25)):
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
    ax0.set_ylim([20,0])
    ax0.set_aspect(10)
    ax0.invert_yaxis()      
    plt.colorbar(cb0, ax = ax0)
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
    ax2.set_ylim([20,0])
    ax2.set_yticks(fre_pos, fre)
    
    ax2.set_aspect(10)
    ax2.invert_yaxis()      
    plt.colorbar(cb2, ax = ax2)
    ax2.set_title('Reconstructed Spectrogram')
             
    fig.tight_layout()
    fig.savefig(f'/home/vsbh19/plots/20 second snippets/DEC_original_embedded_reconst_{files[filenum][:-3]}_{stationname}_{duration}_C{n_clusters}_{idx}.png')
    plt.show()

sys.exit(1) #Completion stamp to shell 
