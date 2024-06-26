#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:12:35 2024

@author: vsbh19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:31:46 2022

@author: stefan and Luke:))
"""
# scheduled data download, formatting and forecasting
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNBadRequestException
import sys
import pandas as pd
#from obspy.clients.iris import Client
client = Client()
# from obspy import read as obread
# from obspy import read_inventory
# from datetime import datetime, timezone
# import time
# import schedule
# import pickle
# import numpy as np

#
def heythem(network, station, **kwargs):
    # check for earthquakes 
    #t_now = datetime.now(timezone.utc)
    #t = UTCDateTime(t_now)
    #tint=120+3600*24 #overlap of one minute to allow for clock drift | need to check for doublets a posteriori
    startt=UTCDateTime("2012-05-22T00:00:01.0")
    endt=UTCDateTime("2022-12-05T23:59:59.0")
    magnitude = 6.0
    maxradius = 9 #teleseism is defined as exceeding more than 1000km or 9 degrees of the earth
    verbose = 2
    stations = [station] if type(station) != list else station #alllows to iterate through one station
    global inventory
    event_ids_set = set() #make sure we dont record the same event
    lats = []; longs = []
    for u in stations: 
        try:
            inv=client.get_stations(network=network,station=u)
            sta=inv[0][0]
            lat=sta.latitude
            lon=sta.longitude
            lats.append(lat)
            longs.append(lon)
        except FDSNNoDataException:
            inventory = client.get_stations(network=network, starttime = startt , endtime= endt)
            print(f"MATE, STATION {u} DOES NOT EXIST AND WILL BE OMITTED HENCEFORTH")
            print("Available stations", inventory)
        except FDSNBadRequestException:
            
            inventory = client.get_stations(starttime = startt, endtime = endt, latitude=36.8817, longitude = 139.4534, maxradius=maxradius, level = "network")
            print("Please make sure the network youre using is correct\n", inventory)
        
            sys.exit()
            
    
    locations = pd.DataFrame(data = {"Long": longs, "Lat": lats})
    
    if "startt" in kwargs:
        startt = UTCDateTime(kwargs['startt'])
    if "endt" in kwargs:
        endt = UTCDateTime(kwargs['endt'])
    if "magnitude" in kwargs:
        magnitude = kwargs["magnitude"]
    if "maxradius" in kwargs: 
        maxradius = kwargs["maxradius"]
    if "filename" in kwargs: 
        filename = kwargs["filename"]
    if "maxradius" in kwargs:
        maxradius = kwargs["maxradius"]
    verbose = kwargs["verbose"] if verbose in kwargs else 2    
    merged_catalog = None
    for i in range(len(stations)):
        try:
            latitude=locations.loc[i, "Lat"]
            longitude = locations.loc[i, "Long"]
            catalog = client.get_events(
                                minmagnitude=magnitude,
                                starttime=startt,
                                endtime=endt,
                                latitude=latitude,
                                longitude=longitude,
                                maxradius=maxradius,
                                includearrivals=True)
            
        except Exception as e:
            if str(e)[0:3] == "Pro":
                print("Fetching events from the data center is proving troublesome. Please check your arguments.")
                sys.exit()
            else:   
                print(str(e)[0:3])
                catalog = None 
                print(f"For station {station[i]} Catalog cannot be retrieved due to unexpected error\n\n\n {e}")
            
        if catalog is not None: 
            if merged_catalog is None: 
                merged_catalog = catalog
            else:
                merged_catalog += catalog
        
    if merged_catalog is not None: 
        try:
            fil=open(f"events_{filename}_mag{magnitude}.txt",'w')
        except: 
            starttt = str(startt); endtt = str(endt) #Utc datetime is not subscriptable
            fil = open(f"eventt_{starttt[:7]}_to_{endtt[:7]}_mag{magnitude}.txt","w")
        for event in merged_catalog:
            evid=str(event.resource_id).split('=')[1]
            evti=str(event).split('\n')[0].split('\t')[1] #time?
            if evid not in event_ids_set:#make sure that we haven't alredy recorded the evebt
                evo=evid+' | '+evti+ ' | '+ str(event.origins[0].depth)+'\n'
                if verbose == 1: 
                    print(event)
                    print(evo)
                    print('-----------------------------------------------------------------------------------------------------\n\n')
                
                    
			     
    
                fil.write(evo)
                event_ids_set.add(evid)
        fil.close()
        
        if verbose == 2:
            print("Returned", len(event_ids_set), "events from\n\n", str(startt)[:-8], "\n------|-------\n", str(endt)[:-8])
        return locations, len(event_ids_set)
    else: 
        sys.exit("GATHERING OF EVENTS HAS FAILED, YOU NEED TO HAVE A LONG HARD LOOK AT YOURSELF")
        
        
#
#events = heythem("IU",["MAJO"], startt="2019-05-22T00:00:01.0",  endt="2022-12-05T23:59:59.0", magnitude = 5.0)
