#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:42:56 2024

@author: vsbh19
"""


""" 
MDS = Multi-dimensional scaling on data 
"""

import sys, os, numpy as np, h5py, matplotlib.pyplot as plt
from sklearn.preprocessing import normalize 
from sklearn.manifold import MDS

#sys.exit()
files = ["ev0000364000.h5", "ev0000593283.h5",
         "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5"]
filenum = 0 
stationname = "IWEH"
directory = os.path.join("/nobackup/vsbh19/wav_files/")
end_day = 30
start_day = 0
component = 2
octave_increase = 6

with h5py.File(directory + f"RawData_{files[filenum][:-3]}_{stationname}_{component}_days{start_day}-{end_day}.h5", "r") as w:
    X = w["data"][:]
    length = len(X)

# X = normalize([X],norm = "max")
# X = np.reshape(X, (length,))

def split(X, chunk): 
    #print("good")
    #global x_ret
    
    #x_ret = np.array(ob)
    data_points_per_chunk = int(len(X)/chunk)
    x_ret = np.empty((chunk,data_points_per_chunk+1)) #file for chunks
    for i in range(0, chunk): #ITERATE DATA AND SPLIT INTO CHUNKS 
        x_ret[i,0] = i*data_points_per_chunk #UPLOADS TIME STAMP FOR BEGINNING OF TIME SERIES
        x_ret[i, 1:] = X[i*data_points_per_chunk:(i+1)*data_points_per_chunk]
        #I NEED TO UPLOAD EACH OF XS POSITIION IN TIME HERE 
    # X_train, X_test = train_test_split(
    #     x_ret, test_size=0.2, shuffle=True, random_state=812) #SPLITS THE TRAINING AND TESTING CHUNKS UP 
    #sys.exit()
    #del x_ret  # need to test what effect this has on the dataset.
    return x_ret 


X = split(X, 1000)
#try:

embedding = MDS(n_components =2, normalized_stress="auto")
X_transformed = embedding.fit_transform(X)
print(X_transformed.shape)
plt.scatter(X_transformed[:,0], X_transformed[:,1])

print("YOU FAILED BITCH")