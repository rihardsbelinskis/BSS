from __future__ import division
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, fftfreq
import numpy as np

n_sampled = 1000
time = np.linspace(0, 10, n_sampled)
comp_1_1 = np.sin(2*10*time)
comp_1_2 = np.cos(2*20*time)
comp_1_3 = np.sin(2*50*time)

y = comp_1_1 + comp_1_2 + comp_1_3
freqs = fftfreq(n_sampled)

mask = freqs > 0
fft_vals = fft(y)
fft_theo = 2*abs(fft_vals/n_sampled)

plt.figure(1),
plt.subplot(2,1,1)
plt.plot(time, y)
plt.title('Original signal')

plt.subplot(2,1,2)
plt.plot(freqs, abs(fft_vals))
plt.title('FFT of original signal')

plt.show()
