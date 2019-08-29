from __future__ import division
import math
import scipy.fftpack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA

"""
===============================================================
A toy problem for Blind Signal Separation using FastICA and FFT
===============================================================
"""

# Generate the total (time-based) signal
n_sampled = 1000
time = np.linspace(0,10, n_sampled)
ptime = np.linspace(0,5, n_sampled/2)

f11 = 1     #1, 5, 50
f12 = 5
f13 = 50

f21 = 4     #4, 20, 100, 27
f22 = 20
f23 = 100
f24 = 27

comp_1_1 = np.sin(2*f11*time)
comp_1_2 = np.cos(2*f12*time)
comp_1_3 = np.sin(2*f13*time)

comp_2_1 = np.sin(2*f21*time)
comp_2_2 = np.sin(2*f22*time)
comp_2_3 = np.sin(2*f23*time)
comp_2_4 = np.sin(2*f24*time)

signal_1 = comp_1_1 + comp_1_2 + comp_1_3 
signal_2 = comp_2_1 + comp_2_2 + comp_2_3 + comp_2_4
tot_signal = np.piecewise(time, [time < 5, time >= 5], [signal_1[:500], signal_2[:500]])

print("Created total signal")
# Separating signals
s1_sep = tot_signal[:int(len(tot_signal)/2)]
s2_sep = tot_signal[int(len(tot_signal)/2):int(len(tot_signal))]
print("Separated into 2 signals")

### Lacks data parsing for a correct format, but once there... ###
S1 = np.c_[comp_1_1, comp_1_2, comp_1_3] # inserting s1_sep
S1 /= S1.std(axis=0)                     # standardaizing the signal

S2 = np.c_[comp_2_1, comp_2_2, comp_2_3, comp_2_4]
S2 /= S2.std(axis=0)

# Mixing data
A1 = np.array([[1,1,1], [0.5, 1.5, 2.0], [1.0, 2.5, 0.5]])
A2 = np.array([[1,1,1,1],[0.5, 1.5, 2.0, 2.5],[1.5, 2.5, 2.0, 0.5],[1, 2.5, 0.5, 0.5]])

X1 = np.dot(S1, A1.T)  # Generate observations
X2 = np.dot(S2, A2.T)

# Computing ICA
ica1 = FastICA(n_components=3)
ica2 = FastICA(n_components=4)

S1_ = ica1.fit_transform(X1)
S2_ = ica2.fit_transform(X2)

# Performing FFT
NFFT = len(s1_sep)
NFFT_pow = 2**(math.ceil(math.log(abs(NFFT),2)))

S1_FFT = scipy.fftpack.fft(S1_)
S1_FFT_noica = scipy.fftpack.fft(s1_sep)
S2_FFT = scipy.fftpack.fft(S2_)
S2_FFT_noica = scipy.fftpack.fft(s2_sep)

XFFT_noica = np.fft.fftfreq(n_sampled, time[1] - time[0])
XFFT_noica = XFFT_noica[:int(len(XFFT_noica)/2)]
XFFT = np.fft.fftfreq(len(S1_FFT), time[1] - time[0])

# Comparing (norm)

# Writing FFT data to file
#pd.options.display.max_rows = 6000
#print(S1_FFT[:len(S1_FFT)])

#fileName = "S1-ICA-params.txt"
#f = open(fileName, 'w')
#f.seek(0)
#f.write(str(S1_))
#f.close()

# ========================================================
# Plotting signals...
#plt.figure(1),
#plt.subplot(3,1,1)
#plt.plot(time, tot_signal)
#plt.title('Total signal')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(3,1,2)
#plt.plot(ptime, s1_sep)
#plt.title('signal 1 separated')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(3,1,3)
#plt.plot(ptime, s2_sep)
#plt.title('signal 2 separated')
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude [-]')
#
#plt.figure(2),
#models = [X1, S1_]
#names = ['Observed signal 1 (after separation)',
#         'ICA recovered signals within signal 1']
#colors = ['red', 'steelblue', 'orange']
#
#for ii, (model, name) in enumerate(zip(models, names), 1):
#    plt.subplot(2, 1, ii)
#    plt.title(name)
#    for sig, color in zip(model.T, colors):
#        plt.plot(time, sig, color=color)
#        plt.ylabel('Amplitude [-]')
#plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
#
#plt.figure(3),
#models = [X2, S2_]
#names = ['Observed signal 2 (after separation)',
#         'ICA recovered signals within signal 2']
#colors = ['red', 'steelblue', 'orange', 'green']
#
#for ii, (model, name) in enumerate(zip(models, names), 1):
#    plt.subplot(2, 1, ii)
#    plt.title(name)
#    for sig, color in zip(model.T, colors):
#        plt.plot(time, sig, color=color)
#
#plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
#plt.ylabel('Amplitude [-]')
#
#plt.figure(3),
#plt.subplot(2,1,1)
#plt.plot(XFFT_noica, abs(S1_FFT_noica))
#plt.title('Signal 1 after FFT (no ICA)')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(2,1,2)
#plt.plot(XFFT_noica, abs(S2_FFT_noica))
#plt.title('Signal 2 after FFT (no ICA)')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.figure(4),
#plt.subplot(2,1,1)
#plt.plot(XFFT, abs(S1_FFT))
#plt.title('Signal 1 after ICA and FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(2,1,2)
#plt.plot(XFFT, abs(S2_FFT))
#plt.title('Signal 2 after ICA and FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.show()
