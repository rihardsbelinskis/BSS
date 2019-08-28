"""
===============================================================
A toy problem for Blind Signal Separation using FastICA and FFT
===============================================================
"""
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA

# Generate the total (time-based) signal
n_sampled = 1000
time = np.linspace(0,10, n_sampled)

f11 = 1
f12 = 5
f13 = 50
f21 = 4
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

# Separating signals

# Performing ICA

# Performing FFT

# Comparing (norm)

# Plotting signals...
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(time, signal_1)
plt.title('signal 1')

plt.subplot(3,1,2)
plt.plot(time, signal_2)
plt.title('signal 2')

plt.subplot(3,1,3)
plt.plot(time, tot_signal)
plt.title('Total signal')

plt.show()
