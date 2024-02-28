#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:57:30 2023

@author: vsbh19
"""
#import tensorflow as tf
#from keras.models import Sequential, Model, load_model
from keras.models import load_model 
#import tensorflow as tf
#from keras import load_model
import os 
import h5py
#import numpy as np
#import matplotlib.pyplot as plt
import sys
#import obspy
date_name = "Nov23"
#dirname = os.path.dirname("/home/vsbh19/initial_python_files/")
nobackupname = os.path.dirname("/nobackup/vsbh19/snovermodels/")

files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5", "Sinosoid.h5"]

try:
    filenum = int(sys.argv[1]) # Which file we want to read.
    stationname = sys.argv[2] #station we want to read
    component = int(sys.argv[3]) #0,1,2 = X,Y,Z
    duration = int(sys.argv[4])
except:
    filenum = 0
    component = 2
    stationname = "KI2H"
    duration = 40
#autoencoder = load_model(nobackupname + f"Saved_Autoencoder_Nov23_{files[filenum][:-3]}_{stationname}.keras")
#encoder = load_model(nobackupname + f"Saved_Encoder1_Nov23_{files[filenum][:-3]}_{stationname}.keras")
#encoder = load_model("/nobackup/vsbh19/snovermodels/Saved_Encoder1_Nov23_ev0000364000_IWEH.keras")
#autoencoder= load_model("/nobackup/vsbh19/snovermodels/Saved_Autoencoder_Nov23_ev0000364000_IWEH.keras")
import h5py
from keras.models import model_from_json

# Specify filenames
encoder_filename = f"Saved_Encoder1_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5"
autoencoder_filename = f"Saved_Autoencoder_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5"

# Construct absolute paths
encoder_path = os.path.join("/nobackup/vsbh19/snovermodels/", encoder_filename)
autoencoder_path = os.path.join("/nobackup/vsbh19/snovermodels/", autoencoder_filename)

encoder = load_model(encoder_path);
autoencoder = load_model(autoencoder_path); 

with h5py.File(f"/nobackup/vsbh19/training_datasets/X_train_X_val_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5" , "r") as f:
    X_train = f.get("X_train")[:]
    #print(X_train[0,:,:,0]); sys.exit()
    X_val = f.get("X_val")[:]
#val_reconst = autoencoder.predict(X_val, verbose = 1) #reconstruction of validation data
val_reconst = autoencoder.predict(X_val, verbose = 1)#; sys.exit()
train_reconst = autoencoder.predict(X_train, verbose = 1)
val_enc = encoder.predict(X_val, verbose = 1)         #embedded latent space samples of validation data
enc_train = encoder.predict(X_train, verbose = 1)     #embedded latent space samples of training data


with h5py.File(nobackupname + f'/Training_LatentSpaceData_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5', 'w') as nf:
    nf.create_dataset('Train_EncodedData', data=enc_train, dtype=enc_train.dtype)
    nf.create_dataset('Val_EncodedData', data=val_enc, dtype=val_enc.dtype)
    nf.create_dataset("Val_ReconstructData", data=val_reconst, dtype = val_reconst.dtype)
    nf.create_dataset("Train_ReconstructData", data=train_reconst, dtype = train_reconst.dtype)

