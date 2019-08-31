from __future__ import division
import csv
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
print("Created total signal...")

# Separating signals
s1_sep = tot_signal[:int(len(tot_signal)/2)]
s2_sep = tot_signal[int(len(tot_signal)/2):int(len(tot_signal))]
print("Separated total signal into 2 signals...")

### Lacks data parsing for a correct format, but once there... ###
S1 = np.c_[comp_1_1, comp_1_2, comp_1_3] # inserting s1_sep
S1 /= S1.std(axis=0)                     # standardaizing the signal

S2 = np.c_[comp_2_1, comp_2_2, comp_2_3, comp_2_4]
S2 /= S2.std(axis=0)

S3 = np.c_[comp_1_1, comp_1_2, comp_1_3, comp_2_1, comp_2_2, comp_2_3, comp_2_4]
S3 /= S3.std(axis=0)

# Mixing data
A1 = np.array([[1,1,1], [0.5, 1.5, 2.0], [1.0, 2.5, 0.5]])
A2 = np.array([[1,1,1,1],[0.5, 1.5, 2.0, 2.5],[1.5, 2.5, 2.0, 0.5],[1, 2.5, 0.5, 0.5]])
A3 = np.array([[1, 1, 1, 1, 1, 1, 1],[7, 6, 5, 4, 3, 2, 1],[2.5, 3.5, 1.0, 4.5, 5.0, 3.0, 1.5],[6, 5, 1.5, 2.5, 1, 4.5, 4.0],[1, 5, 7, 2, 3, 5, 4.5],[0.5, 1.5, 2.5, 7, 6.5, 6, 4],[7, 3, 3.5, 1, 5.5, 1.5, 6]])

X1 = np.dot(S1, A1.T)  # Generate observations
X2 = np.dot(S2, A2.T)
X3 = np.dot(S3, A3.T)

# Computing ICA
ica1 = FastICA(n_components=3)
ica2 = FastICA(n_components=4)
ica3 = FastICA(n_components=7)

S1_ = ica1.fit_transform(X1)
S2_ = ica2.fit_transform(X2)
S3_ = ica3.fit_transform(X3)
print("Computed ICA of signal 1 and signal 2...")

# Performing FFT of all signals (not individualizing S params)
S1_FFT = scipy.fftpack.fft(S1_)
S1_FFT_noica = scipy.fftpack.fft(s1_sep)
S2_FFT = scipy.fftpack.fft(S2_)
S2_FFT_noica = scipy.fftpack.fft(s2_sep)
S3_FFT = scipy.fftpack.fft(S3_)
S3_FFT_noica = scipy.fftpack.fft(tot_signal)

XFFT_noica = np.fft.fftfreq(n_sampled, time[1] - time[0])
XFFT_noica = XFFT_noica[:int(len(XFFT_noica)/2)]
XFFT = np.fft.fftfreq(len(S1_FFT), time[1] - time[0])

XFFT_tot = np.fft.fftfreq(len(S3_FFT), time[1] - time[0])
XFFT_tot_noica = scipy.fftpack.fftfreq(len(tot_signal), time[1] - time[0])
print("Computed FFT of signal 1, signal 2 and total signal...")

# Isolating each S component in signal 1, performing FFT individually
sparams_S1 = S1_
csvFile = open('Signal1_S_params.csv', 'w')
with csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(sparams_S1)
print("S params of signal 1 written to a .csv.")

S_1_all = []
S_2_all = []
S_3_all = []

file = open("Signal1_S_params.csv", "r")
reader = csv.reader(file)
for line in reader:
    S_val1 = line[0]
    S_val2 = line[1]
    S_val3 = line[2]
    S_1_all.append(S_val1)
    S_2_all.append(S_val2)
    S_3_all.append(S_val3)

S1_v1_FFT = scipy.fftpack.fft(S_1_all)
S1_v2_FFT = scipy.fftpack.fft(S_2_all)
S1_v3_FFT = scipy.fftpack.fft(S_3_all)

XFFT_s1 = np.fft.fftfreq(len(S1_v1_FFT), time[1] - time[0])

# Isolating each S component in signal 2, performing FFT individually
sparams_S2 = S2_
csvFile2 = open('Signal2_S_params.csv', 'w')
with csvFile2:
    writer2 = csv.writer(csvFile2)
    writer2.writerows(sparams_S2)
print("S params of signal 2 written to a .csv.")

S2_1_all = []
S2_2_all = []
S2_3_all = []
S2_4_all = []

file = open("Signal2_S_params.csv", "r")
reader2 = csv.reader(file)
for line in reader2:
    S2_val1 = line[0]
    S2_val2 = line[1]
    S2_val3 = line[2]
    S2_val4 = line[3]
    S2_1_all.append(S2_val1)
    S2_2_all.append(S2_val2)
    S2_3_all.append(S2_val3)
    S2_4_all.append(S2_val4)

S2_v1_FFT = scipy.fftpack.fft(S2_1_all)
S2_v2_FFT = scipy.fftpack.fft(S2_2_all)
S2_v3_FFT = scipy.fftpack.fft(S2_3_all)
S2_v4_FFT = scipy.fftpack.fft(S2_4_all)

XFFT_s2 = np.fft.fftfreq(len(S2_v1_FFT), time[1] - time[0])

# Isolating each S component in total signal, performing FFT individually
sparams_S3 = S3_
csvFile3 = open('Signal3_S_params.csv', 'w')
with csvFile3:
    writer3 = csv.writer(csvFile3)
    writer3.writerows(sparams_S3)
print("S params of total signal written to a .csv.")

S3_1_all = []
S3_2_all = []
S3_3_all = []
S3_4_all = []
S3_5_all = []
S3_6_all = []
S3_7_all = []

file = open("Signal3_S_params.csv", "r")
reader3 = csv.reader(file)
for line in reader3:
    S3_val1 = line[0]
    S3_val2 = line[1]
    S3_val3 = line[2]
    S3_val4 = line[3]
    S3_val5 = line[4]
    S3_val6 = line[5]
    S3_val7 = line[6]
    S3_1_all.append(S3_val1)
    S3_2_all.append(S3_val2)
    S3_3_all.append(S3_val3)
    S3_4_all.append(S3_val4)
    S3_5_all.append(S3_val5)
    S3_6_all.append(S3_val6)
    S3_7_all.append(S3_val7)

S3_v1_FFT = scipy.fftpack.fft(S3_1_all)
S3_v2_FFT = scipy.fftpack.fft(S3_2_all)
S3_v3_FFT = scipy.fftpack.fft(S3_3_all)
S3_v4_FFT = scipy.fftpack.fft(S3_4_all)
S3_v5_FFT = scipy.fftpack.fft(S3_5_all)
S3_v6_FFT = scipy.fftpack.fft(S3_6_all)
S3_v7_FFT = scipy.fftpack.fft(S3_7_all)

XFFT_s3 = np.fft.fftfreq(len(S3_v1_FFT), time[1] - time[0])

# Comparing (norm)

# Writing FFT data to file
#pd.options.display.max_rows = 6000
#print(S1_FFT[:len(S1_FFT)])

#fileName = "S1-ICA-params.txt"
#f = open(fileName, 'w')
#f.seek(0)
#f.write(str(S1_))
#f.close()
#print("Written ICA params to a file...")

# =====================PLOTTING===========================
# Plotting signals...
#print("Plotting on screen...")

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
#plt.figure(5),
#plt.subplot(3,1,1)
#plt.plot(XFFT_s1, abs(S1_v1_FFT))
#plt.title('S1 (signal 1) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(3,1,2)
#plt.plot(XFFT_s1, abs(S1_v2_FFT))
#plt.title('S2 (signal 1) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(3,1,3)
#plt.plot(XFFT_s1, abs(S1_v3_FFT))
#plt.title('S3 (signal 1) after ICA and FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.figure(6),
#plt.subplot(4,1,1)
#plt.plot(XFFT_s2, abs(S2_v1_FFT))
#plt.title('S1 (signal 2) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(4,1,2)
#plt.plot(XFFT_s2, abs(S2_v2_FFT))
#plt.title('S2 (signal 2) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(4,1,3)
#plt.plot(XFFT_s2, abs(S2_v3_FFT))
#plt.title('S3 (signal 2) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(4,1,4)
#plt.plot(XFFT_s2, abs(S2_v4_FFT))
#plt.title('S4 (signal 2) after ICA and FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.figure(7),
#plt.subplot(7,1,1)
#plt.plot(XFFT_s3, abs(S3_v1_FFT))
#plt.title('S1 (total signal) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(7,1,2)
#plt.plot(XFFT_s3, abs(S3_v2_FFT))
#plt.title('S2 (total signal) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(7,1,3)
#plt.plot(XFFT_s3, abs(S3_v3_FFT))
#plt.title('S3 (total signal) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(7,1,4)
#plt.plot(XFFT_s3, abs(S3_v4_FFT))
#plt.title('S4 (total signal) after ICA and FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(7,1,5)
#plt.plot(XFFT_s3, abs(S3_v5_FFT))
#plt.title('S5 (total signal) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(7,1,6)
#plt.plot(XFFT_s3, abs(S3_v6_FFT))
#plt.title('S6 (total signal) after ICA and FFT')
#plt.ylabel('Amplitude [-]')
#
#plt.subplot(7,1,7)
#plt.plot(XFFT_s3, abs(S3_v7_FFT))
#plt.title('S7 (total signal) after ICA and FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.figure(8),
#models = [X3, S3_]
#names = ['Observed total signal (no separation)',
#         'ICA recovered signals within total signal']
#colors = ['red', 'steelblue', 'green', 'yellow', 'orange', 'brown', 'purple']
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
#plt.figure(9),
#plt.plot(XFFT_tot_noica, abs(S3_FFT_noica))
#plt.title('Total signal after FFT (no ICA)')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.figure(10),
#plt.plot(XFFT_tot, abs(S3_FFT))
#plt.title('Total signal after ICA and FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Amplitude [-]')
#
#plt.show()
# =====================PLOTTING===========================
