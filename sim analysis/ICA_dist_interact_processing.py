import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fftpack
from scipy import signal, arange

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
Fs = 686
time = np.linspace(0, 8, n_samples)

#s1 = pd.read_csv('S0_dist_inter.csv')  
#s2 = pd.read_csv('S1_dist_inter.csv')
#s3 = pd.read_csv('S2_dist_inter.csv')

s1 = np.sin(2*3.14+5)
s2 = np.cos(3*3.14-2)
s3 = np.sin(5*3.14+15)

S = np.c_[s1, s2, s3]
#S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
print('Mixing matrix:')
print(A_)

##print('Storing reconstructed signals in a .csv file')
##fileName = "S_params_for_FFT.csv"
##with open(fileName, mode='w') as output:
##    output = csv.writer(fileName, 

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
#pca = PCA(n_components=3)
#H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# AS A CHECK: performing FFT, FFT^(-1) and ICA^(-1)
F = scipy.fftpack.fft(S_)
invFFT = scipy.fftpack.ifft(F)
invICA = ica.inverse_transform(invFFT.real)    # ICA does not accept imag. values



# #############################################################################
# Plotting results

print('Plotting results...')
plt.figure(1)
models = [X, S_, invICA]
names = ['Observed interaction signals (mixed signal)',
         'ICA recovered signals',
         'After FFT, inv(FFT) and inv(ICA) -> original interaction signals']
colors = ['red', 'steelblue', 'orange']
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
