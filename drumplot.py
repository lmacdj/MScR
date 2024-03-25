#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:40:10 2024

@author: vsbh19
"""
from obspy.core.trace import Trace, Stats, UTCDateTime; import numpy as np; import sys
import matplotlib.pyplot as plt 
from physicsmetrics import physics_metrics
# class drumplot:
#     def __init__ (self, length, n_clusters, start_time, end_time, **kwargs):
#         self.length = length *100 #assumes 100Hz dataset
#         self.n_clusters = n_clusters 
    
#     # def __str__(self):
#     #     return "Creating drumplot"
def drumplot(data, labels, length, n_clusters, start_time, end_time, station_num, **kwargs):
    day_diff = 3600*24
    end_of_series_time = UTCDateTime("2012-05-22T00:00:01.0")
    clipping = False
    
    if "end_of_series_time" in kwargs:
        end_of_series_time = UTCDateTime(kwargs['end_of_series_time'])
    if "clipping" in kwargs: 
        clipping = kwargs["clipping"]
    if "colors" in kwargs: 
        colors = kwargs["colors"]
    else: 
        colors = np.random.rand(n_clusters,3)
    amount = kwargs.get("amount", 20)
    spec_length = kwargs.get("spec_length", 240)
    number_rows = kwargs.get("nrows", 4)
    calculate_energy = kwargs.get("energy", True)
    
    component = "single"
    #component = kwargs["component"] if component in kwargs else 1
    section = data
    #if component == 1:
    # stats = Stats()
    # stats.sampling_rate  = 100
    # stats.starttime = 
    # 
    # stats
    
    if clipping == True:
        section= np.clip(section, -np.max(section)/15, np.max(section)/15)
        
    section_Trace = Trace(data = section)
    section_Trace.stats.sampling_rate = 100
    section_Trace.stats.starttime = end_of_series_time - (end_time - start_time)*day_diff
    section_Trace.stats.endttime = end_of_series_time
    print(section, end_of_series_time)
    #section.plot()

    section_Trace.plot(type = "dayplot", handle = True, show = True)
    
    sorted_times = labels.sort_values(by="Time", ascending = True)
    sorted_times = sorted_times.loc[sorted_times["Station"] == station_num]
    #print(sorted_times, start_time, end_time)
    sorted_times = sorted_times.loc[sorted_times["Time"] >= start_time]
    sorted_times = sorted_times.loc[sorted_times["Time"] <= end_time]
    #sorted_times = np.asarray((sorted_times))
    #print(sorted_times)#
    current = sorted_times["Label"].iloc[0] #CURRENT LABEL
    counter_old = 1 #CURRENT TIME 
    alpha = 0.05
    color_sections = []
    time_old = section_Trace.stats.starttime 
    q_old = min(sorted_times["Time"])#start time to append to 
    done = set() #all labels that have been presented 
    rows , cols = number_rows,n_clusters
    fig, axes = plt.subplots(nrows = rows, ncols=n_clusters, sharey=True)
    
    # for p,q in enumerate(sorted_times["Time"]): #iterate throigh all the times
       
    #     if sorted_times["Label"].iloc[p] != current: # IF LABEL DIFFERS
    #         #REQUIREMENT TO MOVE ON LABEL HAS BEEN ACHIEVED SO ANOTHER COLOR WILL BE PLOTTED
    #         current_label = int(sorted_times["Label"].iloc[p])
    #         if p % amount == 0 or current_label not in done: #print out only every 20 iterations 
    #             section_Trace_original = section_Trace.copy() 
    #             difference = (q-q_old)*day_diff
    #             tto = time_old + difference 
    #             print(time_old, tto)# type(time_old), type(tto), q*day_diff); sys.exit()
    #             section_Trace.trim(time_old,tto) #time for the time section we want
    #             #section_Trace.plot()
    #             fig = plt.figure()
    #             ax = fig.add_subplot(1, 1, 1)
    #             ax = axes[counter_old % (rows-1), current_label-1]
    #             ax.set_title(f"Trace for Label {str(current)}\nTotal Time {str(tto-time_old)}")
    #             ax.plot(section_Trace.times("matplotlib"), section_Trace.data, color = colors[current_label])
    #             ax.xaxis_date()
    #             fig.autofmt_xdate()
    #             plt.show()
    #             section_Trace = section_Trace_original #revert back to originl
    #             done.add(current_label)
                
    #         """FORMAT (start, end, color for label) """
    #         # axes[2].axvspan(counter_old*240+80, 
    #         #             q*240+80, alpha = alpha, color = colours[current],  ymin = 0.1,  label = f"Label {current}")
    #         counter_old = p
            
    #         time_old += int((q - q_old)*day_diff)
    #         q_old = q
    #         current = sorted_times["Label"].iloc[p] 
    #         #sys.exit()
    cluster_instances = {}
    for label in range(n_clusters):
        instances = sorted_times[sorted_times["Label"] == label]
        #try:
        cluster_instances[label] = instances.sample(min(len(instances), rows)) #random_state = 812) #sample certain instances of each cluster
        
    for label, instances in cluster_instances.items(): #for each label and its instances
        #instance_old = min(instances["Time"])#start time to append to     
        for i, (_, instance) in enumerate(instances.iterrows()):
            #difference = (instance-instance_old)*day_diff
            section_Trace_original = section_Trace.copy()
            #print(time_old)
            section_start_time = time_old + (instance["Time"] - start_time)*day_diff
            section_end_time = time_old + (instance["Time"] - start_time)*day_diff + spec_length
            #print(section_start_time, section_end_time)
            #next_instance_time = instances.iloc[i + 1]["Time"] if i + 1 < len(instances) else end_time
            #instance_duration = 
            section_Trace.trim(section_start_time, section_end_time) / 1000
            ax = axes[i, label] if len(cluster_instances[label]) > 1 else axes[label]
            
            ax.plot(section_Trace.times("matplotlib"), section_Trace.data, color=colors[label])
            #ax.xaxis_date()
            
            #fig.autofmt_xdate()
            
            if calculate_energy:
                metrics = physics_metrics(section_Trace.data, unit = "kilo")
                ax.text(-0.5, 0.5, f'E{metrics.timeseries_energy()}',
                        horizontalalignment='left', fontsize = "xx-small",
                        verticalalignment='top', transform=ax.transAxes,
                        bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                # ax.text(-0.5, 1, f'RMS{metrics.rms()}',
                #         horizontalalignment='left', 
                #         verticalalignment='top', transform=ax.transAxes,fontsize = "xxsmall",
                #         bbox = dict(boxstyle='round', facecolor='olive', alpha=0.5))
            section_Trace = section_Trace_original
            
    print("Shown examples of labels", done)
    #fig.set_title(f"Example timeseries for {station_num}")
    
    plt.tight_layout()
    plt.tick_params(axis = "x",
                    bottom = False)
    plt.tick_params(axis = "y",
                    left = False)
    plt.show()
    #plt.savefig(f"/home/vsbh19/plots/Clusters_station/{files[filenum]}/{files[filenum][:-3]}_{u}_{component}_{duration}_{norm}_Cluster{n_clusters}_Examples.png")
    return fig
    