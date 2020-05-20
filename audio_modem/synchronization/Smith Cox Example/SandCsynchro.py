import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

rec_data = np.genfromtxt("ofdm_data3.csv",dtype=complex, delimiter=",")
PREFIX_LEN = 512
## The prefix is prepended twice but we use its original length for estimation

def calcP_R_M(rx_signal, L):
    """
    Parameters
    ----------
    rx_signal : LIST
        The received signal prepended by the preamble twice
        (Tested with added delay and noise)

    L : INT
        Preamble length
        
    Returns
    -------
    Pr : LIST
        P metric as described in Schmidl & Cox paper
    Rr : LIST
        R metric as described in Schmidl & Cox paper
    M  : LIST
        M metric as described in Schmidl & Cox paper

    """
    rx1 = rx_signal[:-L]
    rx2 = rx_signal[L:]
    mult = rx1.conj() * rx2
    square = abs(rx1**2)
    
    a_P = (1, -1)
    b_P = np.zeros(L)
    b_P[0] = 1 
    b_P[-1] = -1
    
    P = scipy.signal.lfilter(b_P, a_P, mult) / L
    R = scipy.signal.lfilter(b_P, a_P, square) / L
    
    Pr = P[L:]
    Rr = R[L:]
    M = abs(Pr/Rr)**2
    return Pr, Rr, M  # throw away first L samples, as they are not correct due to filter causality

Pr,Rr,M = calcP_R_M(rec_data,PREFIX_LEN)   #Calculates preamble starting point by finding a plateu just before the preamble starts 

plt.plot(abs(Pr), 'b--', lw=3, label='$P(d)$ (equation (6))'); 
plt.plot(abs(Rr), 'r--', lw=3, label='R, method 1')
plt.show()

plt.plot(abs(rec_data), label='$r[n]$', color='cyan')
plt.plot(M, label='$M(d)$')
plt.xlim(0,2000)
plt.show()


plt.subplot(211)
plt.plot(abs(rec_data))

plt.subplot(212)
#Filter to turn plateau into an peak for detection 
b_toPeak = np.ones(PREFIX_LEN) / PREFIX_LEN
a = (1,)
M_filt = scipy.signal.lfilter(b_toPeak, a, M)
    

plt.plot(M,label='M(d) Metric')
#plt.plot(M_filt)


#Differentiate the filtered data
D = np.diff(M_filt)

zeroCrossing_2 = ((D[:-1] * D[1:]) <= 0) * (M[1:-1] > 0.01)

b_ignore = np.ones(1+1024+512) 
b_ignore[0] = 0  
ignore_times = (scipy.signal.lfilter(b_ignore, (1, ), zeroCrossing_2) > 0).astype(int)
zeroCrossing_3 = zeroCrossing_2 * (ignore_times == 0)   # keep only the zero-crossings where the ignore-window is not on

for i in range(len(zeroCrossing_3)):
    if zeroCrossing_3[i]:
        print(i)

plt.plot(zeroCrossing_3, label='Preamble Start')
# plt.legend(bbox_to_anchor=(0.63,0.33),
#            bbox_transform=plt.gcf().transFigure)
plt.show()


#We have used the first block of data as our preamble so the M metric finds a large match with both the preamble and the first block
#Conditioning the signal to ignore matches less than the OFDM length means we only detect the preamble



