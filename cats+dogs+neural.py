#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:39:53 2023

@author: vsbh19
"""

#from cats+dogs.py import x_train, y_train
import sys
directory = "/home/vsbh19/initial_python_files/"
sys.path.append(directory + "cats_dogs.py") #adds cats and dogs to the current path
from cats_dogs import x_train, y_train, x_val, y_val
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D , MaxPooling2D 
#####################GPU OPTIONS############################
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.666)
sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
############################################################
x_train = x_train/255 #normalises data between 0 and 1 
x_train = np.array(x_train)
y_train = np.array(y_train)
#x_train = np.ndarray.tolist(x_train)
model = Sequential() #start of model

model.add(Conv2D(64, (3, 3), input_shape = x_train.shape[1:])) #64 node and receptive field is 3 by 3 ??? input shape = skips the first 
#-1 in reshaping it in the previous code 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3, 3))) #64 window pane and receptive field is 3 by 3 ??? input shape = skips the first 
#-1 in reshaping it in the previous code 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64)) #flatten as dense layer requires 1d representation 
model.add(Activation("relu"))

model.add(Dense(1)) #layer for classifying images 
model.add(Activation("sigmoid"))

model.compile(optimizer = "adam", loss="binary_crossentropy", metrics =["accuracy"])

model.fit(x_train, y_train, batch_size = 8, epochs = 10, validation_split = 0.1)
# predictions = model.predict([x_val])
# print(np.argmax(predictions[1]))