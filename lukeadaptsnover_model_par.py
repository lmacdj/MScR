#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:51:21 2023

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
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5"]
filenum = 1
train_dataname = f"/nobackup/vsbh19/h5_files/Spectrograms_{files[filenum][:-3]}_training.h5"
# with h5py.File(train_dataname, "r") as f:
with h5py.File(train_dataname, "r") as f:
    # m = 2000 #number of samples
    # fre = f.get("frequencies")[:] #frequencies
    # ti = f.get("times")[:]#time windows = 61
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                            samplewise_center=True,
                                            samplewise_std_normalization=True)
    #X_train.extend(X_train)
    #print(X_train)
    
    
    X = f["Data"]; 
    n,o,p = X.shape; 
    fre_scale = X.dims[1]["Frequency scale"]
    ti_scale =X.dims[0]["Time scale"]
    #print(fre_scale, ti_scale); sys.exit()
    #X = np.reshape(X, (len(fre_scale), len(ti_scale),1))
    #fre_mesh, ti_mesh = np.meshgrid(fre_scale, ti_scale)
    #Sxx = X[:,:,0]
    #fre_scale = np.reshape(fre_scale, (o,1)); ti_scale = np.reshape(ti_scale, (1,p))
    
    plt.pcolormesh(ti_scale, fre_scale, np.log10(np.array(X[0,:,:])), shading='auto')
    print(X.shape)
    
    print(np.asarray(X).shape)
    
    X = np.reshape(X, (n,o,p,1))
    datagen.fit(X)
    X = datagen.standardize(X) 
    ti_len = len(ti_scale); fre_len = len(fre_scale)
    X_train, X_val = train_test_split(X, test_size=0.2, 
                                      shuffle=True, 
                                      random_state=812)  
    # X_train = X_train.reshape(len(X_train), ti_len,fre_len, 1)
    # X_val = X_val.reshape(len(X_val), ti_len, fre_len, 1)
    #X_train = X
    #X_val = X_train
    del X
    #datagen.fit(X_val)

################################################MODEL VARIABLES ########################################
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


#model = Sequential()
#model = Cropping2D((0,6),(0,1))(img_input)
###############################################CROPPING USING CONVOLUTIONAL AUTOENCODER ###################################################
#model = Cropping2D(cropping = ((0, 6), (0, 1)))(img_input) #crops the model for image input 
model = Cropping2D(cropping = ((0, 0), (0, 13)))(img_input) #crops the model for image input 
#model = Sequential()
model = Conv2D(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (model)
model = Conv2D(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (model)
model = Conv2D(depth*2**2, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same') (model) 
model = Conv2D(depth*2**3, (3,3), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding ='valid')(model)

shape_before_flattening  = K.int_shape(model)
X = Flatten()(model)
##############################################ENCODING###############################################
encoded = Dense(latent_dim, activation = activation, name ="encoded")(X)


###############################DECODING#####################
d = Dense(np.prod(shape_before_flattening[1:]), activation = "relu")(encoded)
d = Reshape(shape_before_flattening[1:])(d) 
d = Conv2DTranspose(depth*2**2, (3,3), strides = strides, activation = activation, kernel_initializer = kernel_initializer , padding = "same")(d)
d = Conv2DTranspose(depth*2**1, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='valid')(d)
d = Conv2DTranspose(depth*2**0, (5,5), strides=strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')(d)

decoded = Conv2DTranspose(1, (5,5), strides = strides, activation = "linear", kernel_initializer=kernel_initializer, padding='same')(d)
####################################################VISUALISING THE MODELS#######################################
autoencoder = Model(inputs = img_input, outputs = decoded, name = "autoencoder")
encoder = Model(inputs = img_input, outputs = encoded, name = "encoder")

autoencoder.summary() 
#sys.exit()
#########################HYPERPARAMETERS ##################################################
LR = 0.00001 
n_epochs = 400
batch_sz = 512 
logger_fname = 'HistoryLog_LearningCurve.csv'.format(date_name)
csv_logger = CSVLogger(logger_fname)

# Early stopping halts training after validation loss stops decreasing for 10 consectutive epochs
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                           mode='min', restore_best_weights=True)

optim = tf.optimizers.Adam(lr=LR)             #Adaptive learning rate optimization algorithm (Adam)
loss = 'mse'                               #Mean Squared Error Loss function

#Compile Encoder & Autoencoder(initialize random filter weights)
def train_model(rank, X_train, X_val): 
    os.environ["CUDA_VISISBLE_DEVICES"] = str(rank)
    encoder.compile(loss=loss,optimizer=optim) 
    autoencoder.compile(loss=loss,
                    optimizer=optim,
                    metrics=[metrics.mae])   

##################################THE RUNNING OF THE CODE#########################################
#print(X_val[:468])# X_train[:4,:468])

    autoencoder.fit(X_train, X_train[:,:140,:204,0], batch_size=batch_sz, epochs=n_epochs, 
                validation_data=(X_val, X_val[:,:140,:204,0]), callbacks=[csv_logger, early_stop])
#############################DISPLAY TRAINIG LOSSS PER EPOCH #######################################################
    hist = np.genfromtxt(logger_fname, delimiter=',', skip_header=1, names=['epoch', 'train_mse_loss', 'train_mae_loss', 'val_mse_loss', 'val_mae_loss'])


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
    
    plt.savefig('/home/vsbh19/initial_python_files/MSE_MAE_Subplots_Loss_Learning_Curve.png')
    plt.show() 
    #saving = os.path.dirname()
    return encoder, autoencoder
    #autoencoder.save(f"/nobackup/vsbh19/snovermodels/Saved_Autoencoder_Nov23_{files[filenum]}.keras")
    #encoder.save(f"/nobackup/vsbh19/snovermodels/Saved_Encoder1_Nov23_{files[filenum]}.keras")
    
    #tf.keras.backend.clear_session()
#val_reconst = autoencoder.predict(X_val, verbose = 1) #reconstruction of validation data
#val_enc = encoder.predict(X_val, verbose = 1)         #embedded latent space samples of validation data
#enc_train = encoder.predict(X_train, verbose = 1)     #embedded latent space samples of training data

if __name__ == "__main__": #prevents the accidental invoking of s script 
    cpunum = mp.cpu_count()    
    X_train_local = np.array_split(X_train, mp.cpu_count()) #split the data equally between the cores
    X_val_local = np.array_split(X_val, mp.cpu_count())
    #sys.exit()
    processes = []
    for batch_train, batch_val in zip(X_train_local, X_val_local): 
        process = mp.Process(target = train_model, args = (cpunum, batch_train, batch_val))
        processes.append(process)
        process.start()
        
    for process in processes:
        process.join()
    
    with h5py.File("/nobackup/vsbh19/X_train_X_val.h5" , "w") as f:
        f.create_dataset("X_train", data= X_train)
        #plt.pcolormesh(X_val[0,:,:,0])
        f.create_dataset("X_val", data = X_val)
        f.flush()
    if not os.path.exists("/nobackup/vsbh19/snovermodels/"):
        os.makedirs("/nobackup/vsbh19/snovermodels/")
    #dirname = os.path.join("/nobackup/vsbh19/snovermodels/")
    
    autoencoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/Saved_Autoencoder_Nov23_{files[filenum]}.keras"))
    encoder.save(os.path.abspath(f"/nobackup/vsbh19/snovermodels/Saved_Encoder1_Nov23_{files[filenum]}.keras"))
    try: 
        autoencoder.flush()
        encoder.flush()
    except: 
        print("doesn't work")
        #pool.starmap(train_model, enumerate(zip(X_train_local, X_val_local))) #I'm assume starmap is how the code interacts?
    #zip combines lists into a single variable
        # pool.close()
        # pool.join() #I'm assume this joins up all the weights ? 
