import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from scipy.io import wavfile

#data = np.genfromtxt("output.txt",dtype='float32', delimiter=",")
rec_freq, data = wavfile.read('speaker.wav')


PREAMBLE_LEN = 512
CYCLIC_PREF = 256
OFDM_BLOCK_LENGTH = 1024  ## This is N
THRESHOLD = 0.1
print(type(data),len(data))

def add_noise(input_data, SNR=1000):
    # Preprocess
    data = input_data
    #data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()
    print(data[1])

    # Add AGWN
    SNR = (10) ** (SNR / 20)
    noise_magnitude = 1 / SNR
    noise = noise_magnitude * np.random.normal(0, 1, len(data))
    noise = noise.tolist()
    data = [datum + noise_datum for datum, noise_datum in zip(data, noise)]
    return np.array(data)


# delay = [0]*1292
# #data = list(data)
# data = np.insert(data,0,delay)
# data = np.array(data)
# print(data[0])
# data = add_noise(data,10)
# print(type(data),len(data))


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
    #data = data[L:-L]
    
    P = [0]*(len(data))
    R = [0]*(len(data))
    M = [0]*len(data)
    
    s=0
    for m in range(L):
       s +=  np.conj(data[m])*data[m+L]
    
    P[0]= s
    
    print(P[0])
    
    s=0
    for m in range(L):
        s += (data[m+L])**2
    
    R[0] = s
    print(R[0])
    
    for d in range(2*L):        
        P[d+1] = P[d] + np.conj(data[d+L])*data[d+2*L] - np.conj(data[d])*data[d+L]
    
    for d in range(2*L,len(P)-1):
          P[d+1] = P[d] + np.conj(data[d-L])*data[d] - np.conj(data[d-2*L])*data[d-L]      
    """
    R-metric:
        Received energy of data. Operation for item d:
            --> add all squared values of items between d+L and d+2L
    # """
    for d in range(2*L):
        R[d+1] = R[d] + abs(data[d+2*L])**2 - abs(data[d+L])**2
    
    for d in range(L,len(R)-1):
        R[d+1] = R[d] + abs(data[d])**2 - abs(data[d-L])**2

    R = np.array(R)    
    energy_threshold = np.sqrt(np.mean(R**2))
    
    
    for d in range(len(M)):
        if R[d] > (energy_threshold):
         M[d] = (abs(P[d])**2)/(R[d]**2) 
        
    plt.figure(1)    
    plt.subplot(211,label='1')
    plt.plot(P,'b',label="P Metric")
    plt.plot(R,'r',label="R Metric")
    plt.title("P,R,M metrics from Schmidl & Cox")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.subplot(212)
    plt.plot(M,'y',label="M metric")
    plt.plot(np.argmax(M),1,'x',label="crude preamble end = "+str(np.argmax(M)))
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.show()
    
    
    return np.array(P), np.array(R), np.array(M)
    
P, R, M = schmidl_cox(data, PREAMBLE_LEN)


def synchro_samples(P,R,M,CP,N):

    # Low Pass Filter to smooth out plateau and noise
    num = np.ones(CP)/CP
    
    den = (1,)
    
    Mf = sg.lfilter(num, den, M)
     
    # plt.subplot(212)    
    # plt.plot(M,label='M Metric')
    # plt.plot(Mf,'r',label = 'Filtered M Metric')
    # plt.show()
    
    #Differentiation turn peaks from the filtered metric into zero crossings
    
    Mdiff = np.diff(Mf)
    
    #plt.subplot(212,label='2')
    plt.figure(2)    
    plt.plot(Mdiff,'r',label = 'Diff Mf Metric')
    # plt.xlim(4230,5780)
    # plt.ylim(0,1)
    plt.show()


    ##Finds All zero crossings that match an M value above a threshold to account for noise
    # Threshold is 0.98, with noise it should be smaller
    
    zero_crossings = ((Mdiff[:-1] * Mdiff[1:])<=0)*(M[1:-1]>THRESHOLD)
   
    ##Multple crossings due to noise. To avoid, after the first crossing we skip the next 
    # N+CP crossings. 
    
    b_ignore = np.ones(1+N+CP) 
    b_ignore[0] = 0  
    ignore_times = (sg.lfilter(b_ignore, (1, ), zero_crossings) > 0).astype(int)
    zeroCrossing_3 = zero_crossings * (ignore_times == 0)   
        
     # for i in range(len(zeroCrossing_3)):
     #     if zeroCrossing_3[i]:
     #         print(i)
    #plt.subplot(212,label='2') 
    plt.plot(zeroCrossing_3, label='Preamble Start')
    plt.show()
    return  [i for i, val in enumerate(zeroCrossing_3) if val] 

    
sample = synchro_samples(P,R,M,CYCLIC_PREF,OFDM_BLOCK_LENGTH)    
    
print(sample)




#plt.plot(zeroCrossing_3, label='Preamble Start')
# plt.legend(bbox_to_anchor=(0.63,0.33),
#            bbox_transform=plt.gcf().transFigure)
#plt.xlim(4700,4710)
#plt.show()


# #We have used the first block of data as our preamble so the M metric finds a large match with both the preamble and the first block
# #Conditioning the signal to ignore matches less than the OFDM length means we only detect the preamble



