import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io.wavfile import read
    
#Modify these with the names of the desired data
# TX_FILE = 'comp_impulse.wav'
# RX_FILE = 'rec_compimp.wav'
TX_FILE = 'chirp.wav'
RX_FILE = 'recorded_chirp.wav'


sample_freq, sample = read(TX_FILE,mmap=False)
data_freq, data = read(RX_FILE, mmap=False)

### Padding the sample to have the same length as the recording
### Needed for correlation

sample = np.concatenate((sample, np.zeros(len(data)-len(sample))))


def lag_finder(sample, data, sample_rate, plot=False, grad_mode = True):
    
    """
    Takes a file to be sent and a received file and finds the difference 
    by which the received file is delayed.
    
    If plot is set, then it will produce a matplotlib plot of the output
    
    Grad Mode: True or False.
    
    If true, it finds the double gradient before correlating. 
    
    If False it just finds correleation between given inputs.
    
    Gradient estimation has yet to be tested fully.
        
    """
    
    n = len(sample)
    
   
    dd_sample = sample
    dd_data = data
 
    if grad_mode: 
    ###Using the second derivative of signals
        dd_sample = np.gradient(np.gradient(sample))
        dd_data = np.gradient(np.gradient(data))
 
    #Correlation between sample and data, normalised
    corr = signal.correlate(dd_data, dd_sample, mode='same') / np.sqrt(signal.correlate(dd_sample, dd_sample, mode='same')[int(n/2)] * signal.correlate(dd_data, dd_data, mode='same')[int(n/2)])
    
    #Create and shift x axis from -0.5 to 0.5
    delay_arr = np.linspace(-0.5*n/sample_rate, 0.5*n/sample_rate, n)
    
    #Estimates the point at which the peak correlation occurs  //This is not robust enough, needs smarter method
    lag = delay_arr[np.argmax(corr)]
    if lag < 0:
        print('data is ' + str(np.round(abs(lag),3)) + 's ahead of the sample')
    else:
        print('data is ' + str(np.round(lag,3)) + ' behind the sample')

    if plot:
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(np.round(lag, 3)) + ' s')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coeff')
        plt.show()

    return lag, dd_data, dd_sample


def synchronisation(sample, data, sample_freq, lag):
     
    """
    This function takes two data sets, their sampled frequency,
    and the lag between them. 
    
    It removes the lag between the beggining of the data (relative 
    to the sample). 
    
    It returns the data with equal length as the sample (for plotting).
    """
    
    ## Lag should be more precise than sample rate
    # Round order of magnitude of lag close to sample precision and add 1 for safety  
    
    lag_sign = lag
    #print(lag,type(lag))
    lag = abs(np.round(lag,int(abs(np.floor(np.log10(abs(lag)))))+1))


    # Find the difference between the two samples
    sample_lag = int(np.floor(lag * sample_freq))
    #Assuming the data is larger and has a delay before the sample is received
    #So remove the sample lag from the data, getting closer to when the sample began
    
    if lag_sign > 0:
        
        data = data[sample_lag:]
        
        #Pad data with zeroes so the lengths of the arrays match
        data = np.concatenate((data, np.zeros(len(sample)-len(data))))
     
    #Assuming the data is smaller than the sample and has a delay after the sample is received
    #So pad data with zeroes in the beginning
    #estimate and remove the end delay
        
    elif lag_sign < 0:
        
        lag= abs(lag)
        data = np.concatenate(((np.zeros(sample_lag)), data))
        end_lag = len(data) - sample_lag
        data = data[:end_lag]
    
    return data






def plot_data(sample,data,sample_freq):
    
    """   
    Various plots: 
    
    1) plots the correlation between the second gradients of sample and data
    
    2) plots the second gradients of the sample and data, along with
    the estimated lag between them
    
    3) plots the data before and after shifting 
    
    4) plots the sample with shifted data
    
    5) Plots the impulse and impulse responce (Relevance?)
    """
    
    delay, gdata, gsample = lag_finder(sample,data,sample_freq,True)
    
    x = np.linspace(0,int(len(gdata)/sample_freq),len(gdata))
    plt.plot(x,gsample/35767,'b',label='sent grad')
    plt.xlabel("Time (s)")
    plt.ylabel("second deriv (N/A)")
    plt.plot(x,gdata,'y',label='rec grad')
    plt.xlabel("Time (s)")
    plt.axvline(delay,color='r',label='lag')
    plt.legend()
    plt.show()
    
    x = np.linspace(0,int(len(data)/sample_freq),len(data))
    plt.plot(x,data,'r', label = 'rec data')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (N/A)")
    plt.title('Before and After Lag shift')
    
    data = synchronisation(sample,data,sample_freq,delay)
    
    plt.plot(x,data,'g',label="shifted rec data")
    plt.legend()
    plt.show()
    
    plt.plot(x,sample/35767,label='sent data')
    plt.plot(x,data,label = 'rec data')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (N/A)")
    plt.legend()
    plt.title('Sent and Received data, matched')
    plt.show()


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axes[0].plot(x, sample/35767)
    axes[1].plot(x, data)
    axes[0].set_ylim(0,1)
    axes[1].set_ylim(0,1)
    axes[0].set_xlim(0,7)
    axes[1].set_xlim(0,7)
    
    axes[1].set_xlabel('Time (s)')
    axes[0].set_xlabel('Time (s)')
    
    axes[1].set_ylabel('Magnitude (Arbitrary)')
    axes[0].set_ylabel('Magnitude (Arbitrary)')
    
    
    axes[0].title.set_text('Impulse')
    axes[1].title.set_text('Response to Impulse')
        
    
    fig.tight_layout()
    return 0

plot_data(sample,data,sample_freq)