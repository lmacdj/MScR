#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:41:39 2023

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
import os 
#print()
directory = os.path.join("/home/vsbh19/initial_python_files")
sys.path.append(directory  + "lukeadaptsnover_clusteringclass.py")
#sys.path.append(directory + "lukeadaptsnover_clusting.py")
from lukeadaptsnover_clusteringclass import ClusteringLayer
#import lukeadaptsnover_clusting
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, BatchNormalization, Flatten, Bidirectional, Dropout, Reshape
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras import optimizers
from keras import metrics
from keras import backend as K 
import random
import h5py

import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5"]
try: 
    filenum = int(sys.argv[1]) # Which file we want to read.
    component = int(sys.argv[3]) #0,1,2 = X,Y,Z
    station = sys.argv[2] #station we want to read
    duration = int(sys.argv[4])
    n_clusters = int(sys.argv[5])
except: 
    filenum = 0
    component = 2
    station = "all"
    duration = 40
    n_clusters = 7
train_dataname = f"/nobackup/vsbh19/h5_files/Spectrograms_{files[filenum][:-3]}_{station}_{component}_{duration}_training.h5"

#n_clusters = 7 #FROM LUKEADAPTSNOVER_CLUSTERING FOR USE WITH GAUSSIAN CLUSTERING 
# with h5py.File(train_dataname, "r") as f:
random.seed(812)
with h5py.File(train_dataname, "r") as f:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                            samplewise_center=True,
                                            samplewise_std_normalization=True)
    X = f["Data"]; 
    n,o,p = X.shape; 
    fre_scale = X.dims[1]["Frequency scale"][:]
    ti_scale =X.dims[0]["Time scale"][:]
    #print(fre_scale, ti_scale); sys.exit()
    #X = np.reshape(X, (len(fre_scale), len(ti_scale),1))
    #fre_mesh, ti_mesh = np.meshgrid(fre_scale, ti_scale)
    #Sxx = X[:,:,0]
    #fre_scale = np.reshape(fre_scale, (o,1)); ti_scale = np.reshape(ti_scale, (1,p))
    
    #plt.pcolormesh(np.log10(np.array(X[12,:,:])), shading='auto'
    
    indices = f["Indices"][:]
    assert len(X)== len(indices), "Length of x should match that of indices"
    
    indices_of_indices = [r for r in range(len(indices))]
    i = indices.shape
    X = np.reshape(X, (n,o,p,1))
    datagen.fit(X)
    X = datagen.standardize(X) #NOT SURE WHETHER TO DO THIS OR NOT 
    ti_len = len(ti_scale); fre_len = len(fre_scale)
    X_train_i, X_val_i = train_test_split(indices_of_indices, test_size=0.2, 
                                      shuffle=True, 
                                      random_state=812)
    X_train_pos = [indices[r] for r in X_train_i] #appends the location in time n
    X_val_pos =[indices[r] for r in X_val_i] #appends the location in time 
    X_train = np.reshape([X[r] for r in X_train_i], (len(X_train_i),o,p,1))
    X_val = np.reshape([X[r] for r in X_val_i], (len(X_val_i),o,p,1))
    #plt.pcolormesh(ti_scale, fre_scale, np.log10(X_train[0,:,:]), shading='auto')
    #sys.exit()
    #X_train = X
    #X_val = X_train
    #sys.exit()
    del X
    

########################################################MODEL VARIABLES ########################################################
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
clustering = ClusteringLayer(n_clusters, name = "clustering")(encoded)

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

#############################################THE FINAL MODEL#################################
#dec_model = Model(inputs = autoencoder.input, outputs = [clustering, autoencoder.output])
dec_model = Model(inputs=autoencoder.input, outputs={"clustering": clustering, "autoencoder": autoencoder.output})
dec_model.summary() 
###########################################################################################
from keras.utils import plot_model
directory = os.path.join("/nobackup/vsbh19/snovermodels/")
DEC_model_fname = f'/home/vsbh19/initial_python_files/DEC_Architecture.png'
plot_model(dec_model, to_file=DEC_model_fname, 
           show_shapes=True)
from IPython.display import Image
Image(filename=DEC_model_fname)
#sys.exit()
##########################################################HYPERPARAMETERS ##################################################
LR = 0.0001 
n_epochs = 400
batch_sz = 512
logger_fname = f'HistoryLog_LearningCurve_DEC_{files[filenum][:-3]}_{station}_{component}_{duration}_{n_clusters}.csv'
csv_logger = CSVLogger(logger_fname)

# Early stopping halts training after validation loss stops decreasing for 10 consectutive epochs
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                           mode='min', restore_best_weights=True)

optim = tf.keras.optimizers.Adam(learning_rate=LR)             #Adaptive learning rate optimization algorithm (Adam)
loss = 'mse'                               #Mean Squared Error Loss function

############################################COMPILE ALL MODELS###########################################################
#encoder.compile(loss=loss,optimizer=optim) 
#autoencoder.compile(loss=loss,
                    #optimizer=optim,
                    #metrics=[tf.keras.metrics.MeanAbsoluteError()]) 
#dec_model.compile(loss = ["kld", loss], loss_weights = [0.1,.9], optimizer = optim)
dec_model.compile(loss={"clustering": "kld", "autoencoder": loss},
                  loss_weights={"clustering": 0.1, "autoencoder": 0.9},
                  optimizer=optim)

######################################################THE RUNNING OF THE CODE############################################
t1 = time.time()
dec_model.fit(X_train, {"clustering": np.ndarray((len(X_train_i), n_clusters)), "autoencoder": X_train[::,136,:40,0]}, batch_size=batch_sz, epochs=n_epochs, 
                 validation_data=(X_val, {"clustering": np.ndarray((len(X_val_i), n_clusters)), "autoencoder": X_val[:,:136,:40,0]}), callbacks=[csv_logger, early_stop])

#############################DISPLAY TRAINIG LOSSS PER EPOCH #######################################################
print(f"Total elapsed time = {time.time() - t1}")
hist = np.genfromtxt(logger_fname, delimiter=',', skip_header=1, names=['epoch', 'train_mse_loss', 'train_mae_loss', 'val_mse_loss', 'val_mae_loss'])

dec_model.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/Saved_DEC_Nov23_{files[filenum][:-3]}_{station}_{component}_{duration}_{n_clusters}.h5"), save_format= "h5") #must be .h5 .keras not compatible
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

plt.savefig('/home/vsbh19/initial_python_files/_DEC_{files[filenum][:-3]}_{station}_{component}_{duration}_{n_clusters}_MSE_MAE_Subplots_Loss_Learning_Curve.png')
plt.show() 
#saving = os.path.dirname()
# if not os.path.exists("/nobackup/vsbh19/snovermodels/"):
#     os.makedirs("/nobackup/vsbh19/snovermodels/")
#dirname = os.path.join("/nobackup/vsbh19/snovermodels/")

#autoencoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/Saved_Autoencoder_Nov23_{files[filenum][:-3]}_{station}_{duration}.h5")) #must be .h5 .keras not compatible
#encoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/Saved_Encoder1_Nov23_{files[filenum][:-3]}_{station}_{duration}.h5"))

#autoencoder.save(f"/nobackup/vsbh19/snovermodels/Saved_Autoencoder_Nov23_{files[filenum]}.keras")
#encoder.save(f"/nobackup/vsbh19/snovermodels/Saved_Encoder1_Nov23_{files[filenum]}.keras")
with h5py.File("/nobackup/vsbh19/X_train_X_val_DEC_{files[filenum][:-3]}_{station}_{component}_{duration}_{n_clusters}.h5" , "w") as f:
    f.create_dataset("X_train", data= X_train)
    #plt.pcolormesh(X_val[0,:,:,0])
    f.create_dataset("X_val", data = X_val)
    f.create_dataset("X_train_pos", data = X_train_pos)
    f.create_dataset("X_val_pos", data = X_val_pos)
    f.flush()



