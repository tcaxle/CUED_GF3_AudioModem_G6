import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io.wavfile import read
    

sample_freq, sample= read('chirp.wav',mmap=False)
data_freq, data = read('recorded_chirp.wav', mmap=False)

### Padding the sample to have the same length as the recording
### Needed for correlation

sample = np.concatenate((sample , np.zeros(len(data)-len(sample))))

DEBUG = False

def lag_finder(y1, y2, sample_rate, plot=False):
    
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sample_rate, 0.5*n/sample_rate, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    if plot:
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coeff')
        plt.show()

    return delay

if DEBUG:
    x = np.linspace(0,int(len(data)/sample_freq),len(data))
    plt.plot(x,data)

#find time lag and round to 7 decimals >> should be smaller than 1/sample_freq
    lag = np.round(lag_finder(sample, data, sample_freq),7)

#difference between the two samples
    sample_lag = int(np.floor(lag * sample_freq))
               
    data = data[sample_lag:]
    data = np.concatenate((data, np.zeros(len(sample)-len(data))))


# zero_lag = lag_finder(sample,data,44100,True)


# x = np.linspace(0,int(len(data)/sample_freq),len(data))
# plt.plot(x,data)
# plt.show()
# plt.plot(x,sample)
# plt.show()