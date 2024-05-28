#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:28:25 2023

@author: vsbh19
"""

"""
Calculating gap statistic = a measure of how spare the training data is. It 
measures how well the dataset can be clustered against a reference distribution 

"""
import h5py
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import os
import time
import sys
#from keras.models import load_model, Model
#from keras.layers import Input, Cropping2D
#from lukeadaptsnover_clusteringclass import ClusteringLayer
#from keras import optimizers
#from lukeadaptsnover_clusteringclass import ClusteringLayer
directory = os.path.join("/nobackup/vsbh19")
files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5", "ev0000447288.h5", "ev0000734973.h5"]

try:
    filenum = int(sys.argv[1]) # Which file we want to read.
    component = int(sys.argv[3]) #0,1,2 = X,Y,Z
    stationname = sys.argv[2] #station we want to read
except:
    filenum = 5
    component = [2]
    stationname = "all"
    duration = 240
    n_clusters = 6
    norm = "l2"
    high_pass = 1
#with h5py.File(directory + f"/snovermodels/Training_LatentSpaceData_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5" , "r") as f:
#    enc_train = f.get("Train_EncodedData")[:]
direct = f"/nobackup/vsbh19/training_datasets/X_train_X_val_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}_C{n_clusters}.h5"
#direct = f"{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_{high_pass}_C{n_clusters}.h5"
with h5py.File(direct , "r") as f:
    print(f.keys())
    X_train = f.get("X_train")[:]
    X_val = f.get("X_val")[:]
    enc_train = f.get("Trained_Encoded_Data")[:]
#date_name = "Nov23"
samples = 2000
latent_space_dim = 14
random_id = np.random.randint(0, len(X_train), samples) #2000 random samples
kmeans_enc = np.zeros([samples,latent_space_dim]) #to be filled with 

for i in range((len(random_id))):
    kmeans_enc[i] = enc_train[random_id[i]] #gets a sample of the training data and computes the GAP STATISTIC

f_min = np.amin(kmeans_enc, axis = 0)
f_max = np.amax(kmeans_enc, axis = 0)
f_mean = np.mean(kmeans_enc, axis = 0)
f_std = np.std(kmeans_enc, axis = 0)

"""
Time to generate a guassian normal distribution and a reference distribution that 
both have the the time parametes (std, means etc) of the training dataset, 
these will be used when testing against clusters
"""

gauss = np.zeros([len(f_mean), len(kmeans_enc)])
uniform = np.zeros([len(f_mean), len(kmeans_enc)])

for i in range(len(f_mean)):
    gauss[i] = np.random.normal(scale = f_std[i], loc = f_mean[i], size = len(kmeans_enc))
    uniform[i] = np.random.uniform(low=f_min[i], high=f_max[i], size=len(kmeans_enc))
    
inertia = [] #sum of squares distance between samples , cluster centroid and centre
diff = []
for i in np.arange(2,20,1):
    model = KMeans(n_clusters = i, n_init = 10,  random_state = 812, verbose = 0).fit(kmeans_enc) #uniform.T ?
    # try:
    #     difference = inertia[i-3] -model.inertia_
    # except: 
    #     difference = 0
    # diff.append(difference)
    inertia.append(model.inertia_)
"""
Hypothesis: less clusters more optimal
"""
fig = plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
plt.title("Inertia plot")
plt.plot(np.arange(2,20,1), inertia, label='Latent feature data')
#plt.plot(np.arange(2,20,1), diff, label = "Differences")
plt.xlabel('Number of Clusters')
ax = fig.gca()
ax.set_xticks(np.arange(0,20,1))
plt.ylabel('Sum-of-Squares Distance')
plt.title("K-means Inertia vs # of Clusters")
plt.legend()
homedir = os.path.join("/home/vsbh19/")
plt.savefig(f'{homedir}/plots/KMeans_Inertia_vs_NumClusters_dec23.png')

plt.show()

tic = time.time()
inertia_gauss = []; sil_gau = []
fig, ax = plt.subplots(9,2, figsize=(15,8))
##################################TRY AND FIT TO GUASSIAN DISTRIBUTION###################
for i in np.arange(2,20,1):
    kmeans_model_gauss = KMeans(n_clusters=i, n_init=10,# precompute_distances=True, 
                                random_state=812).fit(gauss.T)
    inertia_gauss.append(kmeans_model_gauss.inertia_)
    q, mod = divmod(i, 2)
    
    vis_gau = SilhouetteVisualizer(kmeans_model_gauss,  ax=ax[q-1][mod])
    vis_gau.fit(gauss.T)
    sci_sil_gau = silhouette_score(gauss.T, kmeans_model_gauss.labels_)
    sil_gau.append(sci_sil_gau)
figl  = plt.figure(56)
plt.plot(range(0,len(sil_gau)), sil_gau)
toc = time.time()
print('Reference Gaussian Distribution KMeans Computation Time : {0:4.1f} minutes'.format((toc-tic)/60)) 

# Repeat this process with the data taken from the uniform reference distribtion
################################TRY AND FIT TO UNIFORM DISTRIBUTION####################
tic = time.time()
inertia_uniform =[]; sil_uni = []
fig, ax = plt.subplots(9, 2, figsize=(15,8))
for i in np.arange(2, 20, 1):
    kmeans_model_uniform = KMeans(n_clusters=i, n_init=10, #precompute_distances=True, 
                                  random_state=812).fit(uniform.T)
    inertia_uniform.append(kmeans_model_uniform.inertia_)
    q, mod = divmod(i, 2)
    vis_uni = SilhouetteVisualizer(kmeans_model_uniform,  ax=ax[q-1][mod])
    vis_uni.fit(uniform.T)
    sci_sil_uni = silhouette_score(uniform.T, kmeans_model_uniform.labels_)
    sil_uni.append(sci_sil_uni)
toc = time.time()
figl  = plt.figure(66)
plt.plot(range(2,len(sil_uni)+2), sil_uni)
print('Reference Unfiform Distribution KMeans Computation Time : {0:4.1f} minutes'.format((toc-tic)/60))  

####################CALCULATE DIFFERENCE IN GAP STATISTIC BETWEEN PLOTS AND REFERENCE DISTRIBUTIONS###################
gap_uniform = np.log(np.asarray(inertia_uniform)) - np.log(np.asarray(inertia)) 
gap_gauss = np.log(np.asarray(inertia_gauss)) - np.log(np.asarray(inertia))

########################Plots for GAP STATISTICS based on GAUSSIAN NORMAL refernce distribution###########
fig = plt.figure(figsize=(20,10))
plt.title("Gaussian reference distribution")
fig.add_subplot(1,2,1)
plt.plot(np.arange(2,20,1), inertia, label='Latent feature data')
plt.plot(np.arange(2,20,1), inertia_gauss, label='Gaussian reference')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum-of-Squares')
plt.title("K-means Inertia vs # of Clusters")
plt.legend()

fig.add_subplot(1,2,2)
plt.plot(np.arange(2,20,1), gap_gauss, label='Gaussian Gap')
plt.xlabel('Number of Clusters')
plt.ylabel('Gap')
plt.title("K-means Gap vs # of Clusters")

plt.savefig(f'{homedir}plots/Kmeans_Gaussian_Gap_dec23.png')

####################Plots for GAP STATISTICS  based on UNIFORM REFERENCE DISTRIBUTION###########

fig = plt.figure(figsize=(20,10))
plt.title("Uniform reference distribution")
fig.add_subplot(1,2,1)
plt.plot(np.arange(2,20,1), inertia, label='Latent feature data')
plt.plot(np.arange(2,20,1), inertia_uniform, label='Uniform reference')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum-of-Squares')
plt.title("K-means Inertia vs # of Clusters")
plt.legend()

fig.add_subplot(1,2,2)
plt.plot(np.arange(2,20,1), gap_uniform, label='Uniform Gap')
plt.xlabel('Number of Clusters')
plt.ylabel('Gap')
plt.title("K-means Gap vs # of Clusters")
plt.legend()

plt.savefig(f'{homedir}plots/Kmeans_Uniform_Gap_dec23.png')

#Save Inertia for each data type to save time later.
with h5py.File(f'{homedir}GapStatistics.hdf5', 'w') as nf:
    nf.create_dataset('inertia_testdata', data=inertia)
    nf.create_dataset('inertia_gauss', data=inertia_gauss)
    nf.create_dataset('inertia_uniform', data=inertia_uniform)

#del feat_min, feat_max, feat_mean, feat_std, gauss, uniform
# del inertia,  model, inertia_gauss, kmeans_model_gauss
# del inertia_uniform, kmeans_model_uniform, gap_gauss, gap_uniform


