#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:33:09 2024

@author: vsbh19
"""
import librosa as lb
import librosa.display
from librosa.effects import pitch_shift
import scipy.io.wavfile as wav
import sys, os, numpy as np, h5py, matplotlib.pyplot as plt
from sklearn.preprocessing import normalize 

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

X = normalize([X],norm = "max")
X = np.reshape(X, (length,))
X_fast = lb.effects.time_stretch(X, rate =2**octave_increase) #speeds up the file ad increases the pitch 
X_fast = pitch_shift(X_fast, sr = 100, n_steps=24)
#sys.exit()
#plt.plot(X_fast)
X_fast = lb.mu_compress(X_fast, mu = 255, quantize=False) #quantize=False)
#mfccs = lb.feature.mfcc(data = X_fast, sr = 100, n_mfcc=13)
speccy = lb.feature.melspectrogram(y = X_fast, sr = 100)
S_dB = librosa.power_to_db(speccy, ref=np.max)
#X_fast*=1000 #volume knob
#plt.plot(X_fast)
wavefile = wav.write(directory +
                     f"Wav_{files[filenum][:-3]}_{stationname}_{component}_days{start_day}-{end_day}_Octave_{octave_increase}.wav", 44100, X_fast.astype(np.float32))

from matplotlib.gridspec import GridSpec


fig = plt.figure(figsize =(10,8))
gs = GridSpec(nrows=2, ncols=1, height_ratios = [1,1])
ax1 = fig.add_subplot(gs[0])

librosa.display.waveshow(X_fast, sr =44100)
ax2 = fig.add_subplot(gs[1])
ax2.set_aspect(0.007)
#librosa.display.specshow(mfccs, sr = 100)
#fig.add_subplot(5,2,1)
img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=44100,fmax=8000)

#plt.colorbar()
