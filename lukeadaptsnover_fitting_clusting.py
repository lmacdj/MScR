#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:09:41 2024

@author: vsbh19
"""

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
from sklearn.cluster import KMeans 
import numpy as np
import h5py
from keras.models import model_from_json
#import obspy
sys.path.append("/home/vsbh19/initial_python_files/lukeadaptsnover_clusteringclass.py")
#sys.path.append(directory + "lukeadaptsnover_clusting.py")
from lukeadaptsnover_clusteringclass import ClusteringLayer
import matplotlib.pyplot as plt
date_name = "Nov23"
#dirname = os.path.dirname("/home/vsbh19/initial_python_files/")
nobackupname = os.path.dirname("/nobackup/vsbh19/snovermodels/")

files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5", "ev0000773200.h5", "ev0000447288.h5", "Sinosoid.h5"]

try:
    filenum = int(sys.argv[1]) # Which file we want to read.
    stationname = sys.argv[2] #station we want to read
    component = [int(sys.argv[3])]  #0,1,2 = X,Y,Z
    duration = int(sys.argv[4])
    n_clusters = int(sys.argv[5])
    norm = sys.argv[6]
except:
    filenum = 0
    component = [2]
    stationname = "all"
    duration = 40
    n_clusters = 6
    norm = "max"
#autoencoder = load_model(nobackupname + f"Saved_Autoencoder_Nov23_{files[filenum][:-3]}_{stationname}.keras")
#encoder = load_model(nobackupname + f"Saved_Encoder1_Nov23_{files[filenum][:-3]}_{stationname}.keras")
#encoder = load_model("/nobackup/vsbh19/snovermodels/Saved_Encoder1_Nov23_ev0000364000_IWEH.keras")
#autoencoder= load_model("/nobackup/vsbh19/snovermodels/Saved_Autoencoder_Nov23_ev0000364000_IWEH.keras")


# Specify filenames
#encoder_filename = f"Saved_Encoder1_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5"
#autoencoder_filename = f"Saved_Autoencoder_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5"
DEC_Filename = f"Saved_DEC_Nov23_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.h5"
# Construct absolute paths
DEC_path = os.path.join("/nobackup/vsbh19/snovermodels/", DEC_Filename)
#autoencoder_path = os.path.join("/nobackup/vsbh19/snovermodels/", autoencoder_filename)

DEC = load_model(DEC_path, custom_objects = {"ClusteringLayer": ClusteringLayer});
###ABOVE ADAPTED FROM https://www.tensorflow.org/guide/keras/serialization_and_saving#registering_the_custom_object
#autoencoder = load_model(autoencoder_path); 

with h5py.File(f"/nobackup/vsbh19/training_datasets/X_train_X_val_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.h5" , "r") as f:
    X_train = f.get("X_train")[:]
    #print(X_train[0,:,:,0]); sys.exit()
    X_val = f.get("X_val")[:]
    Labels = f.get("Labels_Train")[:]
    labels_last = f.get("Labels_Last_Train")[:]
    X_test = f.get("X_test")[:]
    
#val_reconst = autoencoder.predict(X_val, verbose = 1) #reconstruction of validation data
#val_reconst = autoencoder.predict(X_val, verbose = 1)#; sys.exit()
#train_reconst = autoencoder.predict(X_train, verbose = 1)
#val_enc = encoder.predict(X_val, verbose = 1)         #embedded latent space samples of validation data
    #embedded latent space samples of training data
""" DEC STUFF
"""
def print_cluster_size(labels):
    """
    Shows the number of samples assigned to each cluster. 
    # Example 
    ```
        print_cluster_size(labels=kmeans_labels)
    ```
    # Arguments
        labels: 1D array  of clustering assignments. The value of each label corresponds to the cluster 
                that the samples in the clustered data set (with the same index) was assigned to. Array must be the same length as
                data.shape[0]. where 'data' is the clustered data set. 
    """
    num_labels = max(labels) + 1
    for j in range(0,num_labels):
        label_idx = np.where(labels==j)[0]
        print("Label " + str(j) + ": " + str(label_idx.shape[0]))
# Parameters for the  DEC finetuning
#--------------------------------------------------------------------------------------------------------------------
batch_size=512                     # number of samples in each batch
tol = 0.001                        # tolerance threshold to stop training
loss = 0                           # initialize loss
index = 0                          # initialize index to start 
maxiter = 10000          # number of updates to rub before halting. (~12 epochs)
update_interval = 315             # Soft assignment distribution and target distributions updated evey 315 batches. 
                                   #(~12 updates/epoch)

#---------------------------------------------STUDENTS T DISTRIBUTION----------------------------------------------

def target_distribution(q):
    """
    Compute the target distribution p, given soft assignements, q. The target distribtuion is generated by giving
    more weight to 'high confidence' samples - those with a higher probability of being a signed to a certain cluster. 
    This is used in the KL-divergence loss function.
    # Arguments
        q: Soft assignement probabilities - Probabilities of each sample being assigned to each cluster.
    # Input:
         2D tensor of shape [n_samples, n_features].
    # Output:
        2D tensor of shape [n_samples, n_features].
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

loss_list = np.zeros([maxiter,3]) 
def fit(X, labels_last, loss, index, index_array):
    global delta_labels, p, q
    delta_labels = []
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            print(ite)
            q, X_reconst  = DEC.predict(X, verbose=2) # Calculate soft assignment distribtuion & CAE reconstructions
            """
            Q = cluster labels 
            train_reconst = reconstructed spectrograms 
            """
            p = target_distribution(q)                      # Update the auxiliary target distribution p       
                
            labels = q.argmax(1)                            # Assign labels to the embedded latent space samples
                
            # check stop criterion - Calculate the % of labels that changed from previous update
            delta_label = np.sum(labels != labels_last).astype(np.float32) /labels.shape[0] 
            delta_labels.append(delta_label)
            labels_last = np.copy(labels)                   # Generate copy of labels for future updates
                
            loss= np.round(loss, 5)                         # Round the loss 
            loss_list[ite, :] = loss
            print('Iter %d' % ite)
            print('Loss: {}'.format(loss))
            print_cluster_size(labels)                      # Show the number of samples assigned to each cluster
                
        
            if ite > 0 and delta_label < tol:               # Break training if loss reaches the tolerance threshhold
                print('BREAKING LOOP: delta_label ', delta_label, '< tol ', tol)
                break
            #Retrain the model 
        idx = index_array[index * batch_size: min((index+1) * batch_size, X.shape[0])]
        loss = DEC.train_on_batch(x=X[idx], y=[p[idx], X[idx,:136,:40,:]])
        index = index + 1 if (index + 1) * batch_size <= X.shape[0] else 0   
        #print(ite)
    plt.plot(np.linspace(0, len(delta_labels), len(delta_labels)), delta_labels)
    plt.plot(len(loss_list), )
    return labels, X_reconst

labels_Xtrain, train_reconst = fit(X_train, labels_last, loss, index, np.arange(X_train.shape[0]))#; sys.exit()
#labels_Xval, val_reconst = fit(X_val, labels_last, loss, index, np.arange(X_val.shape[0]))
#labels_Xtest, test_reconst = fit(X_test, labels_last, loss, index, np.arange(X_test.shape[0]))

with h5py.File(nobackupname + f'/DEC_Training_LatentSpaceData_{files[filenum][:-3]}_{stationname}_{component}_{duration}_{norm}_C{n_clusters}.h5', 'w') as nf:
    #nf.create_dataset('Train_EncodedData', data=enc_train, dtype=enc_train.dtype)
    #nf.create_dataset("Train_Rec", data = reconst)
    #nf.create_dataset('Val_EncodedData', data=val_enc, dtype=val_enc.dtype)
    #nf.create_dataset("Val_ReconstructData", data=val_reconst, dtype = val_reconst.dtype)
    
    nf.create_dataset("Train_ReconstructData", data=train_reconst, dtype = train_reconst.dtype)
    nf.create_dataset("Labels_Train", data = labels_Xtrain, dtype = labels_Xtrain.dtype)
    
    #nf.create_dataset("Test_ReconstructData", data=test_reconst, dtype = test_reconst.dtype)
    #nf.create_dataset("Labels_Test", data=labels_Xtest, dtype = labels_Xtest.dtype)
sys.exit(1)