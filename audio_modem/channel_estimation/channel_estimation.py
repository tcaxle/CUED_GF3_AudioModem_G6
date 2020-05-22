import numpy as np
import matplotlib.pyplot as plt



def channel_estimation(received_data, sent_data, int(N), int(CP):
    """
    received_data: array of received symbols
    sent_data: array of sent symbols
    N: Block length
    CP: Cyclic Prefix size

    h: array of the first K elements of the responce in the time domain
    """
    
    data_length = len(sent_data)
                       
    if(data_length % (N+CP) != 0):           
        raise ValueError("Block length is not a whole number")
    if (data_length != len(received_data)):
        raise ValueError("Arrays have different lengths")

   

    H = []
    num_blocks = int(data_length / (N + CP))

    for i in range(0, num_blocks):
        start = i*(N+CP) + CP 
        end = lower_index + N

        y_block = sent_data[start : end]
        x_block = received_data[start : end]

        DFT_y = np.fft.fft(y_block, N)
        DFT_x = np.fft.fft(x_block, N)

        H_datum = np.true_divide(
            DFT_y, 
            DFT_x,
            out=np.zeros_like(DFT_y), #returns an array of zeros with the same shape
            where=DFT_x!=0
        )
        H.append(H_datum)

    # take average
    # H = np.average(H, axis=0)
    
    h = np.fft.ifft(H, N)
    #h = np.real_if_close(h)

    h = h[ : K + 1] # concatenate to length K
    return h




def plot_in_time(h_time):
    plt.figure()
    plt.title("Time Response")
    plt.xlabel("Sample")
    plt.ylabel("H")
    plt.plot(h_time)
    plt.show()




def plot_in_freq_domain(h_time, N):
    H_freq = np.fft.dft(h_arr, N)
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Frequency Response")
    gain = np.abs(H_freq)
    ax1.plot(gain)
    ax1.set(ylabel="Gain")

    phase = np.unwrap(np.angle(H_freq, deg=True))
    ax2.plot(phase)
    ax2.set(ylabel="Phase")

    plt.show()
