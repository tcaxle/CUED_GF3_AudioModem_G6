import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

##K is arbitrary and determines the coefficients of channel impulse responce, MAX = CP

def channel_estimation(received_data, sent_data, N, CP,K=49):
    """
    Parameters
    ----------
    received_data : Numpy Array of float32
        Array of received data
        
    sent_data : Numpy Array of float32
        Array of transmitted data 
    N : INT
        OFDM block length
    CP : INT
        Cyclic prefix length
    K : INT
        Cut-off point for channel response coefficients

    Returns
    -------
    h : LIST
        The coefficients of the channel impulse responce
    """
    
    N = int(N)
    CP = int(CP)
    
    data_length = len(sent_data)
                       
    if(data_length % (N+CP) != 0):           
        raise ValueError("Block length is not a whole number")
    if (data_length != len(received_data)):
        raise ValueError("Arrays have different lengths")

   

    H = []
    num_blocks = int(data_length / (N + CP))

    for i in range(0, num_blocks):
        start = i*(N+CP) + CP 
        end = start + N

        y_block = sent_data[start : end]
        x_block = received_data[start : end]

        DFT_y = np.fft.fft(y_block, N)
        DFT_x = np.fft.fft(x_block, N)

        H_datum = np.true_divide(DFT_x, DFT_y,out=np.zeros_like(DFT_y), where=DFT_x!=0)
        H.append(H_datum)

    # Take average value of H determined for each block
    H = np.average(H, axis=0)
    
    h = np.fft.ifft(H, N)

    h = h[ : K + 1] # Find the first K coefficients
    return h



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

    plt.show()



########## TESTING ##########
#channel_response = [1, 0.5, 0.35, 0.23, 0.2, 0.3, 0]
# data = np.genfromtxt('sent_data.txt',dtype='float32',delimiter=',')
# data = data[4410:15002]
# convolved_signal = sg.convolve(data, channel_response)
# convolved_signal = convolved_signal[:-(len(channel_response)-1)]
# h = channel_estimation(convolved_signal,data,1024,300)
# print(abs(np.round(h,4)))
# plot_in_freq_domain(h,1024)