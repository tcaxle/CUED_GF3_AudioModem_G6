import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

data = np.genfromtxt("output.txt",dtype='float32', delimiter=",")
PREAMBLE_LEN = 512
CYCLIC_PREF = 300
OFDM_BLOCK_LENGTH = 1024  ## This is N
THRESHOLD = 0.5

## The prefix is prepended twice but we use its original length for estimation

def schmidl_cox(data,L):
    
    
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
    
    for d in range(len(M)-2*L):
        if R[d] != 0:
          M[d] = (abs(P[d])**2)/(R[d]**2) 
        else:
            M[d] = 0
            
    # plt.subplot(211)   
    # plt.plot(P,'b',label="P Metric")
    # plt.plot(R,'r',label="R Metric")
    # plt.subplot(212)
    # plt.plot(M,'y',label="M metric")
    # plt.legend()
    # plt.show()
    
    ##### Remove None values here
    P = [datum for datum in P if datum != None]
    
    R = [datum for datum in R if datum != None]
    
    M = [datum for datum in M if datum != None]
    
    return np.array(P), np.array(R), np.array(M)
    
P, R, M = schmidl_cox(data, PREAMBLE_LEN)


def synchro_samples(P,R,M,CP,N):

    # Low Pass Filter to smooth out plateau and noise
    num = np.ones(CP)/CP
    
    den = (1,0)
    
    Mf = scipy.signal.lfilter(num, den, M)
     
    # plt.subplot(212)    
    # plt.plot(M,label='M Metric')
    # plt.plot(Mf,'r',label = 'Filtered M Metric')
    # plt.show()
    
    #Differentiation turn peaks from the filtered metric into zero crossings
    
    Mdiff = np.diff(Mf)
    
    plt.plot(Mdiff,'r',label = 'Diff Mf Metric')
    plt.xlim(4230,5780)
    plt.show()


    ##Finds All zero crossings that match an M value above a threshold to account for noise
    # Threshold is 0.98, with noise it should be smaller
    
    zero_crossings = ((Mdiff[1:] * Mdiff[:-1])<=0)
   
    zero_crossings = zero_crossings*(M[1:-1]>0.98)
  
    ##Multple crossings due to noise. To avoid, after the first crossing we skip the next 
    # N+CP crossings. 
    
    "PLEASE CLEAN UP"
    
    for i in range(len((zero_crossings))):
        if zero_crossings[i] == True:
            for j in range(i+1,N+CP+i+1):
                zero_crossings[j] = False
    
    return  [i for i, val in enumerate(zero_crossings) if val] 

    
sample = synchro_samples(P,R,M,CYCLIC_PREF,OFDM_BLOCK_LENGTH)    
    
print(sample)




# plt.plot(zeroCrossing_3, label='Preamble Start')
# # plt.legend(bbox_to_anchor=(0.63,0.33),
# #            bbox_transform=plt.gcf().transFigure)
# plt.xlim(4700,4710)
# plt.show()


# #We have used the first block of data as our preamble so the M metric finds a large match with both the preamble and the first block
# #Conditioning the signal to ignore matches less than the OFDM length means we only detect the preamble



