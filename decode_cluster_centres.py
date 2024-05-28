#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:31:31 2024

@author: vsbh19
"""
import tensorflow as tf
from keras.models import load_model
import os
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; import h5py 
from sklearn.cluster import KMeans
import sys
files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5","ev0000447288.h5", "ev0000734973.h5", "Sinosoid.h5"]
filenum = 5
component = [2]
stationname = "all"
duration = 240
start_day =0
end_day = 31.25
norm = "l2"
n_clusters = 6
high_pass = 1
decoder_filename = f"Saved_Decoder_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}.h5"
encoder_filename = f"Saved_Encoder_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}.h5"
decoder_path = os.path.join("/nobackup/vsbh19/snovermodels/", decoder_filename)
os.chdir("/nobackup/vsbh19/snovermodels/")
decoder = load_model(decoder_path)
encoder = load_model(encoder_filename)
#sys.exit()
centre_name = f"CLUSTER_CENTRES_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}_C{n_clusters}.csv"
path_add_cluster = f"{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}_C{n_clusters}"
test_dataname = f"/nobackup/vsbh19/h5_files/TESTING_Spectrograms_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}.h5"
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



datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        samplewise_center=True,
                                        samplewise_std_normalization=True)
with h5py.File(test_dataname, "r") as nn:
    X_test = nn["Data"][:]
    print(nn.keys())
    n,o,p = X_test.shape
    #Test_indices = n["Indices"][:]
    #Test_stations = n["Stations"][:]
    X_test = np.reshape(X_test, (n,o,p,1))
    X_test = datagen.standardize(X_test)
    try:
        X_test_pos = nn["Indices"][:]
        X_test_stations = nn["Stations"][:]
    except: 
        nn.get("Indices")[:]
        nn.get("Stations")[:]
enc_test = encoder.predict(X_test)
kmeans = KMeans(n_clusters=n_clusters, n_init = 100) #n_init = number of initizialisations to perform
labels_test = kmeans.fit_predict(enc_test)
labels_last_test = np.copy(labels_test)
testing =1 
with h5py.File(f"/nobackup/vsbh19/training_datasets/X_TEST_{path_add_cluster}.h5", "w") as nf: 
    nf.create_dataset("X_test", data = X_test)
    nf.create_dataset("Labels_Test", data = labels_test)
    nf.create_dataset("Labels_Last_Test",data = labels_last_test)
    nf.create_dataset("Test_Encoded_Data", data = enc_test)
    nf.create_dataset("X_test_pos", data = X_test_pos)
    nf.create_dataset("X_test_station", data = X_test_stations)
# except:
#     print("NO TESTING DATA AVAILABLE")
#     testing = 0 