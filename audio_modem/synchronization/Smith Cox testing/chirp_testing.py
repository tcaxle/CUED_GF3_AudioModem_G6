"""
A simple script to play a sound of a certain frequency for a duration
"""
DEBUG = False

import numpy as np
import scipy.signal as sg
from scipy.io.wavfile import read
import matplotlib.pyplot as plt


##Importing the Wav Files
sample_freq, sample= read('chrip.wav',mmap=False)
data_freq, data = read('recorded_chirp.wav', mmap=False)

#Normalise  (happened during generation, can be removed from there if needed)
sample = [i/32767 for i in sample] 

if DEBUG:
    print(data[0:300])

#Convert to Freq Domain
    
spectrum_rec_data = np.fft.fft(data)    
spectrum_sample_data = np.fft.fft(sample)
    
if DEBUG:
    print(sample[0:20])
    print(spectrum_sample_data[:20])
    print(data[200:240])
    print(spectrum_rec_data[:20])
  
    
    
#### Plotting Frequency Domain
    
N1 = 220500   #Sample Frequency * Duration
N2 = 132300
# sample spacing
T = 1.0 / 44100
x1 = np.linspace(0.0, N1*T, N1)
x2 = np.linspace(0.0,N2*T,N2)
xf1 = np.linspace(0.0, 1.0/(2.0*T), N1//2)
xf2 = np.linspace(0.0,1.0/(2.0*T),N2//2)
fig, ax = plt.subplots()
ax.plot(xf2, 2.0/N2 * np.abs(spectrum_sample_data[:N2//2]))
ax.plot(xf1, 2.0/N1 * np.abs(spectrum_rec_data[:N1//2]))
plt.xlim(0,3000)
plt.show()

####



####Cross-Correlation in the Freq Domain    
    
## 1) Zero padding = data length + sample length  - 1
#p = len(sample) + len(data) - 1
#sample = np.pad(sample, (0, p), 'constant')
#data = np.pad(data, (0, p), 'constant')

##Make sample the same length as recording
l = len(data)-len(sample)
sample = np.pad(sample, (0, l),'constant')

## 2) Convert to Freq Domain
    
spectrum_rec_data = np.fft.fft(data)    
spectrum_sample_data = np.fft.fft(sample)


## 3) Find the conjugate of the recorded data spectrum
spectrum_rec_data = np.conjugate(spectrum_rec_data)

print(len(spectrum_rec_data))
print(len(spectrum_sample_data))

correlated = np.multiply(spectrum_sample_data,spectrum_rec_data)

corr = np.fft.ifft((correlated))

plt.plot(corr)
plt.show()

    
#### Result is Identical in the 
    
""" 
Failed correlation attempt #2    
    
# overall_pearson_r = np.corrcoef(y[0],y[1])
# print(f"computed Pearson r: {overall_pearson_r}")
# # out: Pandas computed Pearson r: 0.2058774513561943

# r, p = stats.pearsonr(y.dropna()['S1_Joy'], y.dropna()['S2_Joy'])
# print(f"Scipy computed Pearson r: {r} and p-value: {p}")
# # out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

# # Compute rolling window synchrony
# f,ax=plt.subplots(figsize=(7,3))
# y.rolling(window=30,center=True).median().plot(ax=ax)
# ax.set(xlabel='Time',ylabel='Pearson r')
# ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}");

# plt.show()

"""


##Generating a chirp instead of using a saved one::
    
# def play_chirp(f0=500,f1=2000,sample_rate=44100, duration=3):
#     # calulcate total number of samples
#     samples = int(sample_rate * duration)

#     # produce numpy array of samples
#     time_array = np.linspace(0, duration, samples, False)

#     # Modulate the array with our frequency
#     note = sg.chirp(time_array,f0,duration,f1,method='linear')

#     # normalize to 16-bit range
#     #note *= 32767 / np.max(np.abs(note))

#     # convert to 16-bit data
#     #note = note.astype(np.int16)

#     # start playback
#     #sd.play(note, sample_rate)

#     # Wait for it to play
#     sd.wait()

#     return note
    
# notes = play_chirp()



#Failed Correlation Attempt #1
    
sample_frequency2, data1= read('chrip.wav',mmap=False)
sample_frequency, data = read('recorded_chirp.wav', mmap=False)

y=data1

y = [i/32767 for i in y]

x = np.linspace(0,5,sample_frequency*5)
x1= np.linspace(0,3,sample_frequency2*3)

plt.plot(x,data)
plt.show()
plt.plot(x1,y)
plt.show() 
#print(notes)
corr = sg.correlate(y,data, mode='full')
plt.plot(corr)
