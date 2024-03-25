#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:31:31 2024

@author: vsbh19
"""

from keras.models import load_model
import os
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt

files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5","ev0000447288.h5", "ev0000734973.h5", "Sinosoid.h5"]
filenum = 5
component = [2]
stationname = "all"
duration = 240
start_day =0
end_day = 31.25
norm = "l2"
n_clusters = 6
decoder_filename = f"Saved_Decoder_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}.h5"

decoder_path = os.path.join("/nobackup/vsbh19/snovermodels/", decoder_filename)
decoder = load_model(decoder_path)
centre_name = f"CLUSTER_CENTRES_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.csv"
centre_path = os.path.join("/nobackup/vsbh19/snovermodels/", centre_name)
centres = pd.read_csv(centre_path)
centres = centres.to_numpy()
back_out = np.empty((n_clusters, 136,40,1))
fig, ax = plt.subplots(nrows = 1, ncols = n_clusters)
for i in range(n_clusters):
    centre_i = centres[:,i+1]
    centre_i = np.reshape(centre_i, (1, 14))
    back_out[i,:,:,:] = decoder.predict(centre_i)
    
    ax[i].pcolormesh(back_out[i,:,:,0])
