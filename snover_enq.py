#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:32:50 2023

@author: vsbh19
"""
directory = "/nobackup/vsbh19/h5_files/"
files = ["ev0000364000.h5",  "ev0000593283.h5",  "ev0000773200.h5",  "ev0002128689.h5",
"ev0000447288.h5",  "ev0000734973.h5",  "ev0001903830.h5"]
import h5py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.layers import Input, Cropping2D, Conv2D
from scipy.signal import spectrogram 
import sys
seconds_per_30_days = 2.592E6
#from keras.layers import ImageDataGenerator
#from keras.models import 
with h5py.File(directory + files[3], "r") as f: 

    shp = f["AS2H"].shape
    X = f["AS2H"][:]
    X = X[0] #reduce to one dimension
    
    
    frequencies, times, Sxx = spectrogram(X, fs = 1000, nperseg = 64, noverlap = 32)
    
    n, o, p = len(frequencies), len(times), len(Sxx) 
    X_new = [frequencies, times, Sxx]; print(X_new) #trying to get it in their format of fequency bis
   
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                            samplewise_center=True,
                                            samplewise_std_normalization=True)
    datagen.fit(X_new)
    print(X_new.shape); sys.exit()
    #standardization bypassed at this stage
    X = datagen.standardize(X_new)
    X_train, X_val = train_test_split(X, test_size=0.2, 
                                      shuffle=True, 
                                      random_state=812) 
    
img_input = Input(shape=(n, o, p))

depth = 8 
strides = 2 #halves each dimension by 2 in each layer - pooling? 
activation = "relu"
kernel_initializer = "glorot_uniform"
latent_dim = 14

e = Cropping2D(cropping = ((0,6), (0,1)))(img_input) #cropping by -6 and -1 in the input
e = Conv2D(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (e)