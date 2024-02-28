#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:04:44 2023

@author: vsbh19
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import sys
import random 
#pip install opencv-python

classes =  ["train", "val"]
sub_classes = ["dog", "cat"]
directory = "/nobackup/vsbh19/archive/"
# for categories in classes:
#     path = os.path.join(directory, categories) #path to train or testing sets
    
#     for sub_categories in sub_classes:
#             path2 = os.path.join(path, sub_categories) #path for cats or dog within path 
#             for img in os.listdir(path2):
#                 img_array = cv2.imread(os.path.join(path2, img), cv2.IMREAD_GRAYSCALE)  #converts each image into a grey numeric array
#                 plt.imshow(img_array, cmap = "gray")
#                 plt.show()
                
def create_training_data(size):
    path = os.path.join(directory, "train") #path to train or testing sets
    training_data = []
    for sub_categories in sub_classes:
            path2 = os.path.join(path, sub_categories) #path for cats or dog within path 
            class_num = sub_classes.index(sub_categories)
            for img in os.listdir(path2):
                try:
                    img_array = cv2.imread(os.path.join(path2, img), cv2.IMREAD_GRAYSCALE)  #converts each image into a grey numeric array
                    new_array = cv2.resize(img_array, (size, size))
                    training_data.append([new_array, class_num])
                except Exception as e: 
                    pass #for if corrupted images exist 
    random.shuffle(training_data) #assigns random order to the training data 
    return training_data

def create_validation_data(size):
    path = os.path.join(directory, "val") #path to train or testing sets
    val_data = []
    for sub_categories in sub_classes:
            path2 = os.path.join(path, sub_categories) #path for cats or dog within path 
            class_num = sub_classes.index(sub_categories)
            for img in os.listdir(path2):
                try:
                    img_array = cv2.imread(os.path.join(path2, img), cv2.IMREAD_GRAYSCALE)  #converts each image into a grey numeric array
                    new_array = cv2.resize(img_array, (size, size))
                    val_data.append([new_array, class_num])
                except Exception as e: 
                    pass #for if corrupted images exist 
    random.shuffle(val_data) #assigns random order to the training data 
    return val_data
training_data, validation_data = create_training_data(50), create_validation_data(50)
size = 50
x_train = []
y_train = []
x_val = []
y_val = []
for features, label in training_data:
    x_train.append(features) #array of the feature data
    y_train.append(label) #array of labelled data ie it 0 or 1 - this is a supevised learning task 

for features, label in validation_data: 
    x_val.append(features)
    y_val.append(label)
x_train = np.array(x_train).reshape(-1, size, size, 1) #x_train must be a numpy array THIS MUST BE DONE FOR KERAS






