import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

K = 100
##Arbitrary that determines the coefficients of channel impulse response, MAX = CP

def channel_estimation(received_data, known_block, N, CP, K):
    
                
    if(len(received_data) % (N+CP) != 0):           
        raise ValueError("Invalid Block Length")


    symbols = [
    
        received_data[i*(N+CP) + CP : (i+1)*(N+CP)] 
    
        for i in range(0,int(len(received_data)/(N+CP)))
    ]


    # Take average value of H determined for each block
    symbols = np.average(symbols, axis=0)
    
    symbols_freq = np.fft.fft(symbols, N)
    
    known_block_freq = np.fft.fft(known_block,N)
    
    channel_response_freq = np.true_divide(symbols_freq, known_block_freq, out=np.zeros_like(symbols_freq), where = known_block_freq !=0)
    
    channel_response = np.fft.ifft(channel_response_freq,N)
    
    channel_response = np.real_if_close(channel_response)
    
    channel_response = channel_response[ : K + 1] # Find the first K coefficients
    
    return channel_response



def plot_in_time(h_time):
    plt.figure()
    plt.title("Time Response")
    plt.xlabel("Sample")
    plt.ylabel("H")
    plt.plot(h_time)
    plt.show()



def plot_in_freq_domain(h_time, N):
    H_freq = np.fft.fft(h_time, N)
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Frequency Response")
    gain = np.abs(H_freq)
    ax1.plot(gain)
    ax1.set(ylabel="Gain")
    
    phase = np.unwrap(np.angle(H_freq, deg=True))
    ax2.plot(phase)
    ax2.set(ylabel="Phase")
    ax2.set(xlabel='Frequency Bins')

    plt.show()


channel_response = [1, 2, 3, 2, 1, 0.5, 0]
data = np.genfromtxt('known_data.txt',dtype='float32',delimiter=',')

known_block = data[-4096:]

convolved_signal = sg.convolve(data, channel_response)

convolved_signal = convolved_signal[:-(len(channel_response)-1)]

h = channel_estimation(convolved_signal,known_block,4096,704,10)
print(abs(np.round(h,4)))
plot_in_freq_domain(h,4096)
plot_in_time(h)