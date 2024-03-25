#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:22:08 2024

@author: vsbh19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:41:04 2023

@author: vsbh19
"""
import sys
import numpy as np
import multiprocessing as mp ##enables parallel computing 
#print()
#sys.path.append(directory  + "lukeadaptsnover.py")
#sys.exit()
directory = "/home/vsbh19/initial_python_files/"
sys.path.append(directory  + "lukeadaptsnover_clusteringclass.py")
#sys.path.append(directory + "lukeadaptsnover_clusting.py")
from lukeadaptsnover_clusteringclass import ClusteringLayer
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, BatchNormalization, Flatten, Bidirectional, Dropout, Reshape
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras import optimizers
from keras import metrics
from keras import backend as K 
from sklearn.cluster import KMeans
import random
import h5py
import os
import pandas as pd; import math

import matplotlib.pyplot as plt #issues with matplotlib on srun?
import time
from sklearn.model_selection import train_test_split

files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5", "ev0000447288.h5", "ev0000734973.h5", "Sinosoid.h5"]

try: 
    filenum = int(sys.argv[1]) # Which file we want to read.
    component = [int(sys.argv[3])] #0,1,2 = X,Y,Z
    station = sys.argv[2] #station we want to read
    duration = int(sys.argv[4])
    n_clusters = int(sys.argv[5])
    norm = sys.argv[6]
    n_epochs = int(sys.argv[7])
    switch = sys.argv[8]
    LR = float(sys.argv[9])
    batch_sz = int(sys.argv[10])
    print("SUCCESSFUL DECLARATION OF VARIABLES")
except: 
    filenum = 0
    component = [2]
    station = "all"
    duration = 240
    n_clusters = 6
    norm = "max"
    n_epochs = 1
    LR = 0.0001 
    batch_sz = 512
    switch = "True"
    print("RUNNING BACKUP VARIABLES")
train_dataname = f"/nobackup/vsbh19/h5_files/Spectrograms_{files[filenum][:-3]}_{station}_{component}_{duration}_{norm}_training.h5"
test_dataname = f"/nobackup/vsbh19/h5_files/TESTING_Spectrograms_{files[filenum][:-3]}_{station}_{component}_{duration}_{norm}.h5"
# with h5py.File(train_dataname, "r") as f:
#SERIOUSLY IMPORTANT VARIABLEs

#-----------------------------------------------------------------------------------------------------------
path_add = f"{files[filenum][:-3]}_{station}_{component}_{duration}_{norm}" #for when there are NO clusters assigned
path_add_cluster = f"{files[filenum][:-3]}_{station}_{component}_{duration}_{norm}_C{n_clusters}"
#=------------------------------------------------------------------------------------------------------------

#END OF SERIOUSLY IMPORTANT VARIABLES
random.seed(812)
with h5py.File(train_dataname, "r") as f:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                            samplewise_center=True,
                                            samplewise_std_normalization=True)
    
    #print(f.keys()); sys.exit()
    try: 
        X = f["Data"][:]; #GET RID OF THIS AFTER DEBUGGING 
    except: 
        X = f["Sinosoid"][:]
    n,o,p = X.shape; #print(X.shape); sys.exit()
    #print(f.keys()); sys.exit()
    try:
        fre_scale = X.dims[1]["Frequency scale"][:]
        ti_scale =X.dims[0]["Time scale"][:]
        indices = f["Indices"][:]
        try:
            stations = f["Stations"][:]
        except:
            stations = np.ones(len(indices))
    except:
        fre_scale = f.get("Frequency scale")[:]
        ti_scale = f.get("Time scale")[:]
        indices = f.get("Indices")[:]
        try:
            stations= f.get("Stations")[:]
        except:
            stations = np.ones(len(indices))
    try: 
    
        truths = f.get("Truths")[:]
        truth_count = 1
    except: 
        truth_count = 0
        print("NO TRUTH DATA MODEL IS UNSUPERVISED")
        
    #print(fre_scale, ti_scale); sys.exit()
    #X = np.reshape(X, (len(fre_scale), len(ti_scale),1))
    #fre_mesh, ti_mesh = np.meshgrid(fre_scale, ti_scale)
    #Sxx = X[:,:,0]
    #fre_scale = np.reshape(fre_scale, (o,1)); ti_scale = np.reshape(ti_scale, (1,p))
    
    #plt.pcolormesh(np.array(X[90,:,:])); sys.exit()
    # plt.pcolormesh(ti_scale, fre_scale, X[100000,:,:], shading='auto')
    # plt.colorbar()
    # sys.exit()
    #indices = X["Indices"]
    print(train_dataname, indices, f.keys())
    assert len(X)== len(indices), "Length of X should match that of Indices"
    assert len(stations) == len(indices), "Length of stations should match that of Indices"
    #sys.exit()
    #sys.exit()2
    indices_of_indices = [r for r in range(len(indices))]
    i = indices.shape
    X = np.reshape(X, (n,o,p,1))
    datagen.fit(X)
    X = datagen.standardize(X) #NOT SURE WHETHER TO DO THIS OR NOT 
    ti_len = len(ti_scale); fre_len = len(fre_scale)
    if files[filenum][:-3] != "Sinosoid": #NON SYNTHETIC DATA 
        X_train_i, X_val_i = train_test_split(indices_of_indices, test_size=0.2, 
                                      shuffle=True, 
                                      random_state=812)
    else: #SYNTEHTIC DATA
        brea_k = int((2*len(X))/3)
        X_train_i = indices_of_indices[0:brea_k]#66.7 % training
        X_val_i = indices_of_indices[brea_k+1:len(X)-1] #33.3% validation data 
        X_train_truths = truths[0:brea_k]
        X_val_truths = truths[brea_k+1:len(X)-1]
        #try:
    X_train_station = [stations[r] for r in X_train_i]
    X_val_station = [stations[r] for r in X_val_i]
    # except: 
    #     X_train_station = station; X_val_station = station
    #     print("THERE IS ONLY ONE STATION")
    X_train_pos = [indices[r] for r in X_train_i] #appends the location in time n
    X_val_pos =[indices[r] for r in X_val_i] #appends the location in time 
    X_train = np.reshape([X[r] for r in X_train_i], (len(X_train_i),o,p,1)) #finds the equivalent index for X_train
    X_val = np.reshape([X[r] for r in X_val_i], (len(X_val_i),o,p,1))
    
    if switch == "True": 
        
        random_pos_flip = np.random.randint(0, int(len(X_train)), int(math.ceil(len(X_train)/2)))
        for i in random_pos_flip:
            X_train[i,:,:,0] = np.flip(X_train[i,:,:,0], 0) #flip along the 0th axis to flip spectorgrams
        print("SPECTROGRAMS SWITCHED")
    print("Training samples ", len(X_train), "\nValidation Samples ", len(X_val))
    
    
    # plt.pcolormesh(X_train[50000,:,:,0])
    # print(X_train[5000,:,:,0]); sys.exit()
    #sys.exit()
    del X
    

###########################################################MODEL VARIABLES #############################################################
#number of frequencies, number of times, number of windows 
#model = Sequential()
img_input = Input(shape = (fre_len, ti_len, 1)) #we're looking for latent features held within frequency
date_name = "Nov23"
depth = 8
strides = 2 #initial depth increases by 2 at each layer
kernel_initializer = "glorot_uniform"
activation = "relu"
latent_dim = 14 
random.seed(812)

###############################################CROPPING USING CONVOLUTIONAL AUTOENCODER ###################################################
#model = Cropping2D(cropping = ((0, 6), (0, 1)))(img_input) #crops the model for image input 
model = Cropping2D(cropping = ((0, 4), (0, 1)))(img_input) #crops the model for image input 
#model = Sequential()
model = Conv2D(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (model)
model = Conv2D(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (model)
model = Conv2D(depth*2**2, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (model) 
model = Conv2D(depth*2**3, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding ='valid')(model)

shape_before_flattening  = model.shape #upgraded syntax 
X = Flatten()(model)
################################################################ENCODING#########################################################
encoded = Dense(latent_dim, activation = activation, name ="encoded")(X)


###############################DECODING#######################################
d = Dense(np.prod(shape_before_flattening[1:]), activation = "relu")(encoded)
d = Reshape(shape_before_flattening[1:])(d) 
d = Conv2DTranspose(depth*2**2, (3,3), strides = strides, activation = activation, kernel_initializer = kernel_initializer , padding = "valid")(d)
d = Conv2DTranspose(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)
d = Conv2DTranspose(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)

decoded = Conv2DTranspose(1, (5,5), strides = strides, activation = "linear", kernel_initializer=kernel_initializer, padding='same')(d)
########################################################VISUALISING THE MODELS#######################################
autoencoder = Model(inputs = img_input, outputs = decoded, name = "autoencoder")
encoder = Model(inputs = img_input, outputs = encoded, name = "encoder")
decoder = Model(inputs = encoded, outputs = decoded, name= "decoder")
autoencoder.summary() 
#sys.exit()

##########################################################HYPERPARAMETERS THAT CAN BE ADJUSTED ##################################################

##############################################################################################################################

logger_fname = f'HistoryLog_LearningCurve_{files[filenum][:-3]}_{station}_{component}_{duration}_{norm}_C{n_clusters}.csv'
csv_logger = CSVLogger(logger_fname)

# Early stopping halts training after validation loss stops decreasing for 10 consectutive epochs
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                           mode='min', restore_best_weights=True)

optim = tf.keras.optimizers.Adam(learning_rate=LR)             #Adaptive learning rate optimization algorithm (Adam)
loss = 'mse'                               #Mean Squared Error Loss function

#Compile Encoder & Autoencoder(initialize random filter weights)
encoder.compile(loss=loss,optimizer=optim) 
autoencoder.compile(loss=loss,
                    optimizer=optim,
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])   
decoder.compile(loss = loss, 
                optimizer = optim, 
                metrics = [tf.keras.metrics.MeanAbsoluteError()])

################################################################THE RUNNING OF THE CODE############################################################
t1 = time.time()
autoencoder.fit(X_train, X_train[:,:136,:40,0], batch_size=batch_sz, epochs=n_epochs,verbose = 2, 
                validation_data=(X_val, X_val[:,:136,:40,0]), callbacks=[csv_logger, early_stop])
##########################################################DISPLAY TRAINIG LOSSS PER EPOCH #######################################################
print(f"Total elapsed time = {time.time() - t1}")
try: 
    hist = np.genfromtxt(logger_fname, delimiter=',', skip_header=1, names=['epoch', 'train_mse_loss', 'train_mae_loss', 'val_mse_loss', 'val_mae_loss'])
    h = True
except: 
    print("Sorry can't generate histogram")
    h = False
    
#---------------------------------------------------------------DEFINE DEC MODEL-------------------------------
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)       #Feed embedded samples to 
model = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output], name = "DEC") #Input: Spectrograms, 
model.compile(loss=['kld',"mse"], loss_weights=[0.1, .9], optimizer=optim) # Initialize model parameters
#autoencoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/Saved_DEC_Nov23_{files[filenum][:-3]}_{station}_{component}_{duration$
#encoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/Saved_DEC_Nov23_{files[filenum][:-3]}_{station}_{component}_{duration}.h5$
#-----------------------------RUN K MEANS CLUSTERING ON ENCODED DATA-----------------------------------
enc_train = encoder.predict(X_train, verbose = 2) 
kmeans = KMeans(n_clusters=n_clusters, n_init = 100) #n_init = number of initizialisations to perform
labels_train = kmeans.fit_predict(enc_train)
labels_last_train = np.copy(labels_train)
#print(kmeans.cluster_centers_)
ap = {}
for i in range(len(kmeans.cluster_centers_)):
    ap[i] = kmeans.cluster_centers_[i]
clusters = pd.DataFrame(ap)
clusters.to_csv(f"/nobackup/vsbh19/snovermodels/CLUSTER_CENTRES_{path_add_cluster}.csv")
print("CLUSTER CENTRES HAVE BEEN EXPORTED")
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_]) 
#----------------------------GET ENCODING FOR TESTING DATA---------------------------------------------------
try:
    with h5py.File(test_dataname, "r") as n:
        X_test = n["Data"][:]
        print(n.keys())
        n,o,p = X_test.shape
        #Test_indices = n["Indices"][:]
        #Test_stations = n["Stations"][:]
        X_test = np.reshape(X_test, (n,o,p,1))
        X_test = datagen.standardize(X_test)
        try:
            X_test_pos = n["Indices"][:]
            X_test_stations = n["Stations"][:]
        except: 
            n.get("Indices")[:]
            n.get("Stations")[:]
    enc_test = encoder.predict(X_test)
    labels_test = kmeans.fit_predict(enc_test)
    labels_last_test = np.copy(labels_test)
    testing =1 
except:
    print("NO TESTING DATA AVAILABLE")
    testing = 0 
###################Save and Show full DEC model architecture##########################
#from keras.utils import plot_model
#DEC_model_fname = f'DEC_CAE_Model_{files[filenum][:-3]}_{station}_{component}.png'
#plot_model(model, to_file=DEC_model_fname, show_shapes=True)
#from IPython.display import Image
#Image(filename=DEC_model_fname)
if h == True:
    plt.figure(figsize=(20,6))
    
    plt.subplot(1,2,2)
    plt.plot(hist['epoch'], hist['train_mae_loss'], label='train_mae_loss')
    plt.plot(hist['epoch'], hist['val_mae_loss'], label='val_mae_loss')
    plt.xlabel('Epochs')
    plt.title('Training Mean Abs. Error')
    plt.legend()
    
    plt.subplot(1,2,1)
    plt.plot(hist['epoch'], hist['train_mse_loss'], label='train_mse_loss')
    plt.plot(hist['epoch'], hist['val_mse_loss'], label='val_mse_loss')
    plt.xlabel('Epochs')
    plt.title('Training Mean Squared Error')
    plt.legend()
    
    plt.savefig(f'/home/vsbh19/initial_python_files/MSE_MAE_Subplots_Loss_Learning_Curve_{files[filenum][:-3]}_{station}_{component}_{duration}_{norm}_C{n_clusters}.png')
    #plt.show() 
#saving = os.path.dirname()
# if not os.path.exists("/nobackup/vsbh19/snovermodels/"):
#     os.makedirs("/nobackup/vsbh19/snovermodels/")
#dirname = os.path.join("/nobackup/vsbh19/snovermodels/")

autoencoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/%sSaved_Autoencoder_Nov23_{path_add}.h5" %("FLIP" if switch == True else "")), save_format= "h5") #must be .h5 .keras not compatible
encoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/%sSaved_Encoder_Nov23_{path_add}.h5"%("FLIP" if switch == True else "")), save_format= "h5")
model.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/%sSaved_DEC_Nov23_{path_add_cluster}.h5" %("FLIP" if switch == True else "")), save_format= "h5")
decoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/%sSaved_Decoder_Nov23_{path_add}.h5" %("FLIP" if switch == True else "")), save_format= "h5")
#autoencoder.save(f"/nobackup/vsbh19/snovermodels/Saved_Autoencoder_Nov23_{files[filenum]}.keras")
#encoder.save(f"/nobackup/vsbh19/snovermodels/Saved_Encoder1_Nov23_{files[filenum]}.keras")
with h5py.File(f"/nobackup/vsbh19/training_datasets/%sX_train_X_val_{path_add_cluster}.h5" %("FLIP" if switch == True else "") , "w") as f:
    f.create_dataset("X_train", data= X_train)
    f.create_dataset("X_val", data = X_val)
    f.create_dataset("Labels_Train", data = labels_train)
    f.create_dataset("Labels_Last_Train", data = labels_last_train)
    f.create_dataset("Trained_Encoded_Data", data = enc_train)
    
    if testing == 1:
        f.create_dataset("X_test", data = X_test)
        f.create_dataset("Labels_Test", data = labels_test)
        f.create_dataset("Labels_Last_Test",data = labels_last_train)
        f.create_dataset("Test_Encoded_Data", data = enc_test)
        f.create_dataset("X_test_pos", data = X_test_pos)
        f.create_dataset("X_test_station", data = X_test_stations)
    #TIME STAMPS 
    f.create_dataset("X_train_pos", data = X_train_pos)
    f.create_dataset("X_val_pos", data = X_val_pos)
    #STATION STAMPS
    f.create_dataset("X_train_station", data = X_train_station)
    f.create_dataset("X_val_station", data = X_train_station)
    f["Frequency scale"] = np.array(fre_scale)
    f["Frequency scale"].make_scale("Frequency scale")
    f["X_train"].dims[1].attach_scale(f["Frequency scale"])
    f["Time scale"] = np.array(ti_scale)
    f["Time scale"].make_scale("Time scale")
    f["X_train"].dims[0].attach_scale(f["Time scale"])
    if truth_count == 1: 
        print("Creating Truth Dataset")
        f.create_dataset("Truths_Train", data = X_train_truths)
        f.create_dataset("Truths_Val", data = X_val_truths)
    f.flush()

sys.exit(1) #Return completion code to shell file 

