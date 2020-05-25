import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

# data = np.genfromtxt("output.txt",dtype='float32', delimiter=",")

# L = PREAMBLE_LEN = 512
# CYCLIC_PREF = 256
# OFDM_BLOCK_LENGTH = 1024  ## This is N
# THRESHOLD = 0.98


## The prefix is prepended twice but we use its original length for estimation

def schmidl_cox(data,preamble_length):
    """
    

    Parameters
    ----------
    data : numpy array of float32 values
    
        An array of received values for which the P,R,M metrics need to be
        calculated for the Schmidl & Cox synchronisation method
    
    preamble_length (aka L): Integer 

        The length of the preamble prepended to the transmitted data. 
        This should be half of the ofdm block length N

    Returns
    -------
    
    P,R,M Metrics: Numpy arrays of float32 values
        Metrics used for synchronisation.
    """
    
    """
    P-metric:
    
    If the conjugate of a sample from the first half (r*_d) is multiplied 
    by the corresponding sample from the second half (r_d+L), the effect of
    the channel should cancel. Therefore, the products of these pairs will be
    very large.
    
    This is an iterative method as described in S&C to create a list of P-metric
    values for the entire received data.
    
    Notes: Misses the last 2L points of data, unsure if this might lead to error
            Not calculating P0 and R0, to save time, assumed irrelevant
    """
    
    L = preamble_length
    P = [None]*(len(data))
    R = [None]*(len(data))
    M = [None]*len(data)
    
    P[0] = 0
    R[0] = 0
    
    for d in range(len(P)-2*L):        
        P[d+1] = P[d] + np.conj(data[d+L])*data[d+2*L] - np.conj(data[d])*data[d+L]
        
    """
    R-metric:
        Received energy of data. Operation for item d:
            --> add all squared values of items between d+L and d+2L
    """
    for d in range(len(R)-2*L):
        R[d+1] = R[d] + abs(data[d+2*L])**2 - abs(data[d+L])**2
    
    """
    M-metric: P squared over R squared

    """

    for d in range(len(M)-2*L):
        if R[d] != 0:
          M[d] = (abs(P[d])**2)/(R[d]**2) 
        else:
            M[d] = 0
            
    #plt.subplot(211)   
    #plt.plot(P,'b',label="P Metric")
    #plt.plot(R,'r',label="R Metric")
    # plt.subplot(212)
    # plt.plot(M,'y',label="M metric")
    # plt.legend()
    # plt.show()
    
    ##### Remove None values here
    P = [datum for datum in P if datum != None]
    
    R = [datum for datum in R if datum != None]
    
    M = [datum for datum in M if datum != None]
    
    return np.array(P), np.array(R), np.array(M)


def snc_start(P,R,M,ofdm_block_length,cp,threshold=0.9):

    # Low Pass Filter to smooth out plateau and noise
    num = np.ones(cp)/cp
    
    den = (1,0)
    
    Mf = sg.lfilter(num, den, M)
     
    #Differentiation turn peaks from the filtered metric into zero crossings
    
    Mdiff = np.diff(Mf)
    

    ##Finds All zero crossings that match an M value above a threshold to account for noise
    # Threshold is 0.98, with noise it should be smaller
    
    zero_crossings = ((Mdiff[:-1] * Mdiff[1:])<=0)*(M[1:-1]>threshold)
   
    ##Multple crossings due to noise. To avoid, after the first crossing we skip the next 
    # N+CP crossings. 
    
    ignored_crossings = np.ones(1+ofdm_block_length+cp) 
    ignored_crossings[0] = 0  
    ignore_times = (sg.lfilter(ignored_crossings, (1, ), zero_crossings) > 0).astype(int)
    zero_crossings = zero_crossings * (ignore_times == 0)   
        

    return  [i for i, val in enumerate(zero_crossings) if val] 

