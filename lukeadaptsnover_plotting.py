#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:37:37 2024

@author: vsbh19
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import h5py
import sys
import os
files = ["ev0000364000.h5","ev0000593283.h5", "ev0001903830.h5", "ev0002128689.h5",  "ev0000773200.h5", "Sinosoid.h5"]
date_name = "Jan24"
try:
    filenum = int(sys.argv[1]) # Which file we want to read.
    component = int(sys.argv[3]) #0,1,2 = X,Y,Z
    stationname = sys.argv[2] #station we want to read
    duration = int(sys.argv[4])
except:
    filenum = 3
    component = 2
    stationname = "all"
    duration = 40
nobackupname = os.path.dirname("/nobackup/vsbh19/snovermodels/")
with h5py.File(nobackupname + f'/Training_LatentSpaceData_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5', 'r') as nf:
    print(nf.keys())
    val_reconst = nf.get("Val_ReconstructData")[:]
    val_enc = nf.get("Val_EncodedData")[:]
    enc_train =nf.get("Train_EncodedData")[:]
    #train_reconst = nf.get("Train_ReconstructData")[:]
    

with h5py.File(f"/nobackup/vsbh19/training_datasets/X_train_X_val_{files[filenum][:-3]}_{stationname}_{component}_{duration}.h5" , "r") as f:
    X_val = f.get("X_val")[:]
    X_train = f.get("X_train")[:]
    fre_scale = f.get("Frequency scale")[:]
    #ti_scale = f.get("Time scale") [:]
secs = ['0', '8', '16', '24', '32']
fre = ["0", "4", "8", "12","16", "20"]
fre_pos = np.arange(0,24,4)
secs_pos = np.arange(0,40,8)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
# Plot in the first subplot
axes[0].set_title("ORIGIN")
#pcm1 = axes[0].pcolormesh((X_val[1000, :120, :40, 0]))
pcm1 = axes[0].imshow(X_val[60, :140, :40, 0], aspect = "auto", cmap = "viridis", origin = "lower")

axes[0].set_ylim([0,16])
axes[0].set_aspect(10)
cbar1 = plt.colorbar(pcm1, ax=axes[0], label='Log Scale')

# Plot in the second subplot
axes[1].set_title("Recon")
pcm2 = axes[1].imshow((val_reconst[60, :140, :40, 0]), aspect = "auto", cmap = "viridis", origin = "lower")
cbar2 = plt.colorbar(pcm2, ax=axes[1], label='Log Scale')
axes[1].set_ylim([0,16])
axes[1].set_aspect(10)
#fig.colorbar()
#sys.exit()
#print(val_reconst[2000,:,:,0])
#plt.show()
#sys.exit()
cnt = 0
fig = plt.figure(figsize=(13,10))
for imgIdx in np.random.randint(0,len(X_val),4):
    cnt = cnt+1
    # Top row shows original input spectrograms
    fig.add_subplot(2,4,cnt)
    plt.imshow(np.reshape(X_val[imgIdx,:136,:40,:], (136,40)))
    #plt.colorbar()
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')
    plt.title(f'Original idx {imgIdx}')
    plt.ylim([0,20])
    # Bottom row shows the reconstructed spectrograms (CAE output)
    fig.add_subplot(2,4,cnt+4)
    plt.imshow(np.reshape(val_reconst[imgIdx], (136,40)))
    #plt.colorbar()
    plt.title(f'Reconstructed idx {imgIdx}')
    plt.gca().invert_yaxis()
    plt.ylabel('Frequency (Hz)')
    plt.xticks(secs_pos, secs)
    plt.xlabel('Time(s)')
    plt.ylim([0,20])
plt.savefig('/home/vsbh19/initial_python_files/Reconstruction_Examples_[].png'.format(date_name))
plt.show()
plt.close()

from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors

for u,idx in enumerate(np.random.randint(0,len(X_val),25)):
    fig = plt.figure(figsize=(8,17))
    gs = GridSpec(nrows=3, ncols=3, width_ratios=[0.1,0.1,0.2], height_ratios = [1,0,1])
    fig.suptitle(f"#{u}, Reconstructed spectrogram number {idx} for {stationname}")
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    eps = 0.5
    #Original spectrogram
    cb0 = ax0.imshow(np.reshape(X_val[idx,:136, :40, :], (136,40)))# norm = colors.LogNorm(vmin = 1e0, vmax = 1e1))
    #c#b0 = ax0.imshow(np.log1p(np.maximum(eps, np.reshape(X_val[idx, :140, :204, :], (140, 204))))
    ax0.set_ylabel('Frequency (Hz)')
    #ax0.set_xticks()
    ax0.set_xticks(secs_pos, secs)
    ax0.set_yticks(fre_pos, fre)
    ax0.set_xlabel('Time(s)')
    ax0.set_ylim([20,0])
    ax0.set_aspect(10)
    ax0.invert_yaxis()      
    plt.colorbar(cb0, ax = ax0)
    ax0.set_title('Original Spectrogram')
    
    #Latent space image representation
    cb1 = ax1.imshow(val_enc[idx].reshape(14,1), cmap='viridis')
    ax1.invert_yaxis()
    ax1.set_aspect(2)
    plt.colorbar(cb1, ax = ax1)
    ax1.set_title('Latent Space')
    
    #Reconstructed Image
    cb2 = ax2.imshow(np.reshape(val_reconst[idx,:, :, :] + eps, (136,40))) #norm = colors.LogNorm(vmin = 1e0, vmax = 1e1))
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xticks(secs_pos, secs)
    ax2.set_xlabel('Time(s)')
    ax2.set_ylim([16,0])
    ax2.set_yticks(fre_pos, fre)
    ax2.set_aspect(10)
    ax2.invert_yaxis()      
    plt.colorbar(cb2, ax = ax2)
    ax2.set_title('Reconstructed Spectrogram')
             
    fig.tight_layout()
    fig.savefig(f'/home/vsbh19/plots/20 second snippets/original_embedded_reconst_{files[filenum][:-3]}_{stationname}_{duration}_{idx}.png')
    plt.show()
