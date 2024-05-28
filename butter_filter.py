#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:36:10 2024

@author: vsbh19
"""
from scipy.signal import butter, filtfilt, freqs
import matplotlib.pyplot as plt
import numpy as np

#%% BUTTER FILTER

def butter_filter(data, btype, cutoff, fs, order, save, **kwargs):
    nyquist_frequency = 0.5*fs
    normal_cutoff = cutoff / (nyquist_frequency)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype=btype, analog=False, **kwargs)
    y = filtfilt(b, a, data)
    if save: 
        w, h = freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(100, color='green') # cutoff frequency
        plt.show()
    return y


# b, a = butter(3, 2/50, "highpass", analog=False)
# w, h = freqs(b,a)
# plt.semilogx(w, 20 * np.log10(abs(h)))
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [radians / second]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')
# plt.axvline(2/(2*np.pi), color='green') # cutoff frequency
# plt.show()