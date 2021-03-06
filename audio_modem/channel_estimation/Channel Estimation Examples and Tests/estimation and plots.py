"""
WE DO NOT NEED THIS FILE BUT I AM KEEPING IT JUST IN CASE
"""

import numpy as np
import scipy.signal as sg
from scipy.io.wavfile import read
import matplotlib.pyplot as plt


##Importing the Wav Files
sample_freq, sample= read('clap.wav',mmap=False)
data_freq, data = read('Charalambos_test.wav', mmap=False)


def organise_blocks(modulated_data, N, CYCLIC_PREFIX) :
    
    block_length = N + CYCLIC_PREFIX
    block_number = int(len(modulated_data) / block_length)
    
    # 1) Split into blocks of N +CP
    modulated_data = np.array_split(modulated_data, block_number)

    # 2) Discard cyclic prefixes (first 32 bits)
    modulated_data = [block[CYCLIC_PREFIX:] for block in modulated_data]

    # 3) DFT N = 1024
    #demodulated_data = np.fft.fft(modulated_data, N)

    return modulated_data


#Convert to Freq Domain
    
spectrum_rec_data = np.fft.fft(organise_blocks(data,1024,512),1024)    
spectrum_sample_data = np.fft.fft(sample,1024)
    

#### Plotting Frequency Domain
        
    
N1 = 1024   #Sample Frequency * Duration
N2 = 1024
# sample spacing
T = 1.0 / 44100
x1 = np.linspace(0.0, N1*T, N1)
x2 = np.linspace(0.0,N2*T,N2)
xf1 = np.linspace(0.0, 1.0/(2.0*T), N1//2)
xf2 = np.linspace(0.0,1.0/(2.0*T),N2//2)
fig, ax = plt.subplots()
yf2 = 2.0/N2 * np.abs(spectrum_sample_data[:N2//2])
yf1 = 2.0/N1 * np.abs(spectrum_rec_data[:N1//2])
ax.plot(xf2, 2.0/N2 * np.abs(spectrum_sample_data[:N2//2]))
ax.plot(xf1, 2.0/N1 * np.abs(spectrum_rec_data[:N1//2]),'r')
plt.xlim(0,10000)
plt.show()


y = np.divide(yf2,yf1)
plt.plot(xf1,y)
#plt.xscale("log")
#plt.yscale("log")
plt.show()

# inversed_channel = np.fft.ifft(y)

# x = np.linspace(0,5,len(data)//2)
# plt.plot(x,inversed_channel)
# plt.show()

