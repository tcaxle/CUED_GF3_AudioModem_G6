"""
A simple script to play a sound of a certain frequency for a duration
"""

import numpy as np
import scipy.signal as sg
from scipy.io.wavfile import read
import matplotlib.pyplot as plt


##Importing the Wav Files
sample_freq, sample= read('comp_Impulse.wav',mmap=False)
data_freq, data = read('rec_compimp.wav', mmap=False)


#Convert to Freq Domain
    
spectrum_rec_data = np.fft.fft(data)    
spectrum_sample_data = np.fft.fft(sample)
    

#### Plotting Frequency Domain
        
    
N1 = len(sample)   #Sample Frequency * Duration
N2 = len(data)
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
plt.yscale("log")
plt.show()

# inversed_channel = np.fft.ifft(y)

# x = np.linspace(0,5,len(data)//2)
# plt.plot(x,inversed_channel)
# plt.show()

