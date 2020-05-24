import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io.wavfile import read,write
    
##### NOTES FOR THE FUTURE: 
##### 1) THIS FILE IS DIFFICULT TO PARSE DUE TO POOR FORMATTING (APOLOGIES)
##### 2) SYNCHRONISATION ONLY WORKS IF DATA DRIFTS FORWARD
##### 3) IF THE DATA DRIFTS BACKWARDS, ESTIMATES GIVE 0 DELAY,
##### SYNCHRO FUNCTION SHIFTS NOTHING
##### 4) CORRELATION DOESN'T WORK IN THAT CASE, BOTH MAX OR GRADIENT MAX FAIL
##### 5) FOR  DELAYS BOTH IN THE BEGGINING AND END OF FILES, 
##### ITERATION THROUGH THE LAG AND SYNCHRO FUNCTIONS SHOULD: CLEAR THEM ALL,
##### THEN ALIGN THE DATA WITH THE SAMPLE AND ADD ZEROES EVERYWHERE ELSE. 
##### THIS WILL BE POSSIBLE IF BACKWARDS DRIFT IS HANDLED, NOT YET IMPLEMENTED
##### 6) THE PLOTTING FUNCTION IS FOR TESTING ONLY, NOT REQUIRED

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
    corr = signal.correlate(dd_data, dd_sample, mode='same')
    
    #This normalised the corr, but it gives errors
    corr = corr / np.sqrt(signal.correlate(dd_sample, dd_sample, mode='same')[int(n/2)] * signal.correlate(dd_data, dd_data, mode='same')[int(n/2)])
    
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
    
    if grad_mode:
        return lag, grad_mode, dd_data, dd_sample 
    else:
        return lag, grad_mode, 0, 0
    
     
def lag_sync(sample, data, sample_freq, lag):
     
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
    
    #Assuming the data has a delay at the beginning
    
    if lag_sign > 0:
                
        #remove the sample lag from the data, getting closer to when the sample began
        data = data[sample_lag:]
        
        #Pad data with sample_lag amount of zeroes so the lengths of the arrays match
        data = np.concatenate((data,(np.zeros(sample_lag,dtype=np.int16))),axis=None)
              
        
    ###Assuming the data arrives faster than the sample (negative delay)
    ###This occurs if we estimate the lag too early. Feeding the signal again should
    ###trigger this and try and shift the data to the right
    
        
    elif lag_sign < 0:
        
        #Pad sample_lag amount of zeroes until data and sample match    
        data = np.concatenate(((np.zeros(sample_lag,dtype=np.int16)), data),axis=None)

        #remove the end samples
        data = data[:-sample_lag]
        
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
    
    delay, grad, gdata, gsample = lag_finder(sample,data,sample_freq,True,grad_mode=True)
    
    # if grad:
    #     x = np.linspace(0,int(len(gdata)/sample_freq),len(gdata))
    #     plt.plot(x,gsample/np.max(gsample),'b',label='sent grad')
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("second deriv (N/A)")
    #     plt.plot(x,gdata/np.max(gdata),'y',label='rec grad')
    #     plt.xlabel("Time (s)")
    #     plt.axvline(delay,color='r',label='lag')
    #     plt.legend()
    #     plt.show()
        
    x = np.linspace(0,int(len(data)/sample_freq),len(data))
    plt.plot(x,data/np.max(data),'r', label = 'rec data')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (N/A)")
    plt.title('Before and After Lag shift')
    
    data = synchronisation(sample,data,sample_freq,delay)
    
    # plt.plot(x,data/np.max(data),'g',label="shifted rec data")
    # plt.legend()
    # plt.show()
    
    # plt.plot(x,sample/np.max(sample),label='sent data')
    # plt.plot(x,data/np.max(data),label = 'rec data')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude (N/A)")
    # plt.legend()
    # plt.title('Sent and Received data, matched')
    # plt.show()


    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # axes[0].plot(x, sample/np.max(sample))
    # axes[1].plot(x, data/np.max(data))
    # axes[0].set_ylim(0,1)
    # axes[1].set_ylim(0,1)
    # axes[0].set_xlim(0,7)
    # axes[1].set_xlim(0,7)
    
    # axes[1].set_xlabel('Time (s)')
    # axes[0].set_xlabel('Time (s)')
    
    # axes[1].set_ylabel('Magnitude (Arbitrary)')
    # axes[0].set_ylabel('Magnitude (Arbitrary)')
    
    
    # axes[0].title.set_text('Impulse')
    # axes[1].title.set_text('Response to Impulse')
        
    
    # fig.tight_layout()
    return data



########    THIS IS ALL TESTING #########
# ####
# TX_FILE = "sent.wav"
# RX_FILE = "beg_rec.wav"   
# sample_freq, sample = read(TX_FILE,mmap=False)
# data_freq, data = read(RX_FILE, mmap=False)
# sample = np.concatenate((sample, np.zeros(len(data)-len(sample))),axis=None)

# lag,grad,dat,dat2 = lag_finder(sample, data, sample_freq,grad_mode=True)

# new_data = synchronisation(sample, data, sample_freq, lag)

# write("beg_rec_fixed.wav",sample_freq,new_data)
# ####


# TX_FILE = "sent.wav"
# RX_FILE = "end_rec.wav"
# sample_freq, sample = read(TX_FILE,mmap=False)
# data_freq, data = read(RX_FILE, mmap=False)
# sample = np.concatenate((sample, np.zeros(len(data)-len(sample),dtype=np.int16)))

# lag,grad,dat,dat2 = lag_finder(sample, data, sample_freq,True,grad_mode=True)

# new_data = synchronisation(sample, data, sample_freq, lag)

# write("end_rec_fixed.wav",sample_freq,new_data)

# TX_FILE = 'sent.wav'
# RX_FILE = "both_rec.wav"
# sample_freq, sample = read(TX_FILE,mmap=False)
# data_freq, data = read(RX_FILE, mmap=False)
# sample = np.concatenate((sample, np.zeros(len(data)-len(sample))))

# lag,grad,dat,dat2 = lag_finder(sample, data, sample_freq,True, grad_mode=True)

# new_data = synchronisation(sample, data, sample_freq, lag)

# write("both_rec_fixed.wav",sample_freq,new_data)
