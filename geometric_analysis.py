#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:10:04 2024

@author: vsbh19
"""
import numpy as np; import pandas as pd
class Geometric_analysis:
    """
    """
    
    def __init__(self, labels, btype, file, n_clusters, station_num,  **kwargs):
        """
        

        Parameters
        ----------
        lat_long_mat : TYPE
            DESCRIPTION.
        lat_long : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        
        self.radius = 6.371E6 
        try:
            self.colors = kwargs.get("colors")
        except: 
            self.colors = ["red", "blue", "green", "purple",  "orange", "black", "cyan", "violet"]
        self.all_labels_master = labels
        self.btype = btype 
        self.file = file
        self.n_clusters = n_clusters
        self.station_num = station_num
    def extract(self):
        """
        

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.

        Returns
        -------
        long : TYPE
            DESCRIPTION.
        lat : TYPE
            DESCRIPTION.

        """
        csv = pd.read_csv(f"/nobackup/vsbh19/Earthquake_Meta/attributes_of_{self.file}.csv")
        print(csv)
        if self.btype == "LatLong":
            lat_long = csv["Latitude", "Longitude"]
            lat_long = lat_long.to_numpy()
            return lat_long
        if self.btype == "Distance_Epicentre":
            distance_epi = csv["Distance from Epicentre"].to_numpy()
            self.distnace_epi = distance_epi
            return distance_epi
    def crowflies(self):
        dims = self.lat_long_mat
        s1, la1 = self.lat_long_epi
        distances = []
        for s2,la2 in dims:
            
            d = self.radius * np.arccos(np.sin(s1)*np.sin(s2) + np.cos(s1)* np.cos(s2) * np.cos(la1-la2)) 
            distances.append(d)
        return distances
    def plot_clusters_distance_time(self, distance_epi, sample_rate, **kwargs):
        
        import matplotlib.pyplot as plt 
        fig,ax = plt.subplots(nrows =1 , ncols=1)
        for i in range(len(self.all_labels_master)):
            distance = distance_epi[i]
            for q in range(0, len(self.all_labels_master[0])):
                
                labels = self.all_labels_master[i][q][0][-1000:]
                ax.scatter(distance*np.ones(len(labels)),labels, color = self.colors[q])
                #ax.tripcolor(distance*np.ones(len(labels)),labels, color = self.colors[q], **kwargs)
                #plots every 10 datapoints
                #print("Label", q)
    def trip_color(self, distance_epi, stations,  times, labels, **kwargs):
        import matplotlib.pyplot as plt 
        fig,ax = plt.subplots(nrows =1 , ncols=1)
        x = stations
        for i in range(len(self.all_labels_master)):
            distance = distance_epi[i]
            print(distance)
            print(i)
            x = np.put(x, i, distance)
        ax.tripcolor(x, times, labels, **kwargs)
        #ax.tripcolor(distance*np.ones(len(labels)),labels, color = self.colors[q], **kwargs)
                #plots every 10 datapoints
    def plot_clusters_distance(self, distance_epi, bound, **kwargs):
        import matplotlib.pyplot as plt 
        fig,ax = plt.subplots(nrows =1 , ncols=1)
        stations_names = [str(i) for i in range(len(self.all_labels_master))]
        
        ratios = {}
        for q in range(0, len(self.all_labels_master[0])):
            
            labels = self.all_labels_master[:][q][1][-bound:]
            frequencies = []
            for i in labels:
                frequency_label = len(i)
                frequencies.append(frequency_label)
                distance = distance_epi[i]
                
                ratios[stations_names[i]] = np.array(frequencies)
                ax.bar(distance,labels, color = self.colors[q], **kwargs)
                
    def stackplot(self,  distance_epi, bound, **kwargs):
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots(nrows =1 , ncols = 1)
        lat_dim = kwargs.get("lat_dim", None)
        try:
            omits = kwargs.get("omits")
            
        except: 
            omits = None
        master = self.all_labels_master
        distance_epi = {index:val for index,val in enumerate(distance_epi)}
        distance_epi = {k: v for k, v in sorted(distance_epi.items(), key=lambda item: item[1])}
        #av = np.empty((len(self.all_labels_master), self.n_clusters))
        av = {}
        for i in range(self.n_clusters):
            averages = []
            
            
            for u in distance_epi.keys():
                if i in omits:
                    
                    averages.append(0)
                    continue
                else:
                    average = np.average(master[u][i][1][-bound:]) #frozen in time !!
                    averages.append(average)
            av[str(i)] = averages
        print(av)
        ax.stackplot(distance_epi.values(),
                     av["0"],
                      av["1"],
                      av["2"],
                      av["3"],
                      av["4"],
                      av["5"], 
                     labels=av.keys())
        ax.set_ylabel("Proportion of cluster")
        fig.suptitle(f"Clusters vs distance eipcentre Lat{lat_dim}")
        ax.legend(loc="upper left", reverse = True)
        #ax.fill_between()
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        ax.set_prop_cycle(color=self.colors)
        ax.set_xlabel("Distance from epicentre (metres)")
                
        