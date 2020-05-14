import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io.wavfile import read
    
#Modify these with the names of the desired data
TX_FILE = 'chirp.wav'
RX_FILE = 'recorded_chirp.wav'


sample_freq, sample = read(TX_FILE,mmap=False)
data_freq, data = read(RX_FILE, mmap=False)

### Padding the sample to have the same length as the recording
### Needed for correlation

sample = np.concatenate((sample, np.zeros(len(data)-len(sample))))

DEBUG = False

def lag_finder(sample, data, sample_rate, plot=False):
    
    """
    Takes a file to be sent and a received file and finds the difference 
    by which the received file is delayed.
    
    If plot is set then it will produce a matplotlib plot of the output
    """
    
    n = len(sample)
    
    #Correlation between sample and data, normalised
    corr = signal.correlate(data, sample, mode='same') / np.sqrt(signal.correlate(sample, sample, mode='same')[int(n/2)] * signal.correlate(data, data, mode='same')[int(n/2)])
    
    #Create and shift x axis from -0.5 to 0.5
    delay_arr = np.linspace(-0.5*n/sample_rate, 0.5*n/sample_rate, n)
    
    #Estimates the point at which the peak correlation occurs  //This is not robust enough, needs smarter method
    lag = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(lag) + ' behind y1')

    if plot:
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(np.round(lag, 3)) + ' s')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coeff')
        plt.show()

    return lag

def synchronisation(lag, sample, data, sample_freq):
     
    """
    This function takes the lag between two data sets, the two data sets
    and the sampled frequency. 
    
    It removes the lag between the beggining of the data (relative 
    to the sample).
    
    It returns the data with equal length as the sample (for plotting).
    """
    
    ## Lag should be more precise than sample rate
    # Round order of magnitude of lag close to sample precision and add 1 for safety    

    lag = np.round(lag,int(abs(np.floor(np.log10(lag))))+1)


    # Find the difference between the two samples
    sample_lag = int(np.floor(lag * sample_freq))
    
    #Assuming the data is always larger and has a delay before the sample is received
    #So remove the sample lag from the data, getting closer to when the sample began
    data = data[sample_lag:]
    
    #Pad data with zeroes so the lengths of the arrays match
    data = np.concatenate((data, np.zeros(len(sample)-len(data))))
    
    return data


##Plots the data before and after shifting
##Then plots the data and sample for comparison
##Primarily for testing so not integrated in the functions


if DEBUG:
    x = np.linspace(0,int(len(data)/sample_freq),len(data))
    plt.plot(x,data)
    
    delay = lag_finder(sample,data,sample_freq)
    
    data = synchronisation(delay,sample,data,sample_freq)

    plt.plot(x,data)
    plt.show()
    plt.plot(x,sample/35767)
    plt.plot(x,data)
    plt.show()