import numpy as np
import scipy.signal as sg
from scipy.io import wavfile
from matplotlib import pyplot as plt

#fig, axs = plt.subplots(4)
N = 4096
CP = 0

def wav_to_binary(input_file="input.wav"):
    """
    Parameters
    ----------
    input_file : FILE NAME (STRING)
        wave data stored as floats in a .wav file

    Returns
    -------
    output_data : STRING
        string binary data
    """

    # Ooen the file and read the data
    output_data = wavfile.read(input_file)[1]
    return output_data
 
def sweep(duration=1, f_start=100, f_end=8000, sample_rate=48000, channels=1):
    """
    Plays a frequency sweep
    """
    # Calculate number of samples
    samples = int(duration * sample_rate)
    # Produce time array
    time_array = np.linspace(0, duration, samples)
    # Produce a frequency awway
    #frequency_array = np.linspace(f_start, f_end, samples)
    # Produce frequency sweep
    f_sweep = sg.chirp(time_array, f_start, duration, f_end)
    # Normalise sweep
    #f_sweep *= 32767 / np.max(np.abs(f_sweep))
    
    #f_sweep = f_sweep.astype(np.int16)
    # Play noise
    # print(f_sweep)
    # recording = sd.playrec(f_sweep, sample_rate, channels=channels)
    # sd.wait()
    return f_sweep, sample_rate, samples
    
def recieve(input_data, known_data):
    
    data = input_data
    data = np.array(data).astype(np.float64)
    known_data = np.array(known_data).astype(np.float64)
    known_data *= 1.0/np.max(np.abs(known_data))
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()
    known_data = known_data.tolist()
    
    dd_sample = np.gradient(np.gradient(known_data))
    dd_data = np.gradient(np.gradient(data))
    #dd_sample = known_data
    #dd_data = data
    sample_rate = 48000
    #Correlation between sample and data, normalised
    corr = sg.correlate(dd_data, dd_sample, mode='valid')
    
    #This normalised the corr, but it gives errors
    #corr = corr / np.sqrt(sg.correlate(dd_sample, dd_sample, mode='same') * sg.correlate(dd_data, dd_data, mode='same'))
    
    #Create and shift x axis from -0.5 to 0.5
    #delay_arr = np.linspace(-0.5*len(known_data)/sample_rate,0.5*len(known_data)/sample_rate,len(known_data))
    
    #Estimates the point at which the peak correlation occurs  //This is not robust enough, needs smarter method
    lag_sample = np.argmax(corr)
    
    lag = []
    for i,datum in enumerate(data):
        if i == lag_sample:
            lag = i
    
    if lag < 0:
        print('data is ' + str(np.round(abs(lag),3)) + 's ahead of the sample')
    else:
        print('data is ' + str(np.round(lag,3)) + ' behind the sample')

 
    # plt.figure()
    # plt.plot(corr)
    # plt.plot(lag)
    # plt.title('Lag: ' + str(lag/sample_rate) + ' s or ' + str(lag) + ' samples')
    # plt.xlabel('Lag')
    # plt.ylabel('Correlation coeff')
    # plt.xlim(0,lag+10000)
    # plt.ylim(0,100)
    # plt.show()
    
    
    data = data[lag+sample_rate:] #remove everything up to the end of the chirp
    #data = data[4096:] #remove first symbol
   
    # plt.plot(data)
    # plt.xlim(0,120000)
    # plt.show()
    #print(data[sample_rate])
    #return lag, dd_data, dd_sample 
    N = 4096
    CYCLIC_PREFIX = 0 
    block_length = N + CYCLIC_PREFIX
    block_number = int(np.ceil(len(data) / block_length))
    
    # 1) Split into blocks of 4096
    modulated_data = np.array_split(data, block_number)
    
    # 3) DFT N = 4096
    demodulated_data = [np.fft.fft(block,N) for block in modulated_data]
    #demodulated_data = np.array_split(demodulated_data,  block_number)
    #print((demodulated_data))
    #print((demodulated_data[3]))
    
    #freq_known_data = np.fft.fft(known_data,N)
    #half_first_block = demodulated_data[1:2048]
    
    #freq_known_data = np.fft.fft(known_data,N)
    
    #channel_resp = demodulated_data[0]/freq_known_data
    
    # plt.plot(channel_resp)
    # plt.show()
    
    # # 4) Convolve with inverse FIR channel
    # # 4.1) Invert the channel
    #inverse_channel = np.fft.fft(channel_resp, N)
    
    # # 4.2) Convolve
    #unconvolved_data = [np.divide(block, channel_resp) for block in demodulated_data]
    # # 4.3) Discard last half of each block
    unconvolved_data = [block[1:2048] for block in demodulated_data]
    
    
    unconvolved_data = np.array(unconvolved_data)
    # 5) Decide on optimal decoder for constellations
    # 5.1) Define constellation
    constellation = {
        complex(+1, +1)/np.sqrt(2): (0, 0),
        complex(-1, +1)/np.sqrt(2): (0, 1),
        complex(-1, -1)/np.sqrt(2): (1, 1),
        complex(+1, -1)/np.sqrt(2): (1, 0),
    }
    # 5.2) Minimum distance decode and map to bits
    plt.figure(2)
    plt.scatter(unconvolved_data[:9].real, unconvolved_data[:9].imag)
    plt.show()
    
    
#     mapped_data = []
#     for block in unconvolved_data:
#         for data_symbol in block:
#         # Get distance to all symbols in constellation
#             distances = {abs(data_symbol - constellation_symbol): constellation_symbol for constellation_symbol in constellation.keys()}
#         # Get minimum distance
#             minimum_distance = min(distances.keys())
#         # Find symbol matching minimum distance
#             symbol = distances[minimum_distance]
#         # Append symbol data to mapped data array
#             mapped_data.append(constellation[symbol][0])
#             mapped_data.append(constellation[symbol][1])

    
#     output_string = ""
#     print(len(mapped_data))
#     for bit in mapped_data:
#         output_string += str(bit)
#         output_data = [output_string[i : i + 8] for i in range(0, len(output_string), 8)]
#     print(output_data[:50])
#     # 6.2) Convert ints to bytearray
#     output_data = bytearray([int(i, 2) for i in output_data])
# # 6.3 Remove first 18 items
#     #output_data = output_data[18:]
#     #del output_data[0:18]
# # 6.4) Write output file
#     with open("output.txt", "wb") as f:
#         f.write(output_data)

    '''
    # Get peaks
    peaks = []
    cutoff = 0.2
    for point in diff:
        if point >= cutoff:
            peaks.append(1)
        else:
            peaks.append(0)
    peak_locations = []

    # Find only peaks that are three samples wide
    for i in range(len(peaks)):
        if peaks[i] == 1 and peaks[i+2] == 1:
            peaks[i+1] = 1
            peak_locations.append(i+1)
        else:
            peaks[i] = 0
    axs[2].plot(diff)
    axs[3].plot(peaks)
    plt.show()

    # Recover blocks
    data = []
    for location in peak_locations:
        data.append(input_data[location - int(N/2) : location + int(N/2)])

    # FFT
    data = [np.fft.fft(block, N) for block in data]

    # Extract peritnent info
    data = [block[1:512] for block in data]

    # Minimum distance map
    #for block in data:
    #    plt.scatter(block.real, block.imag)
    #plt.show()
    for i in range(len(data)):
        block = data[i]
        output = []
        for datum in block:
            distances = {abs(datum - value) : key for key, value in CONSTELLATION.items()}
            min_distance = min(distances)
            output.append(distances[min_distance])
        data[i] = output

    # Flatten and join to one string
    data = [datum for block in data for datum in block]
    data = "".join(data)

    # Write data
    data = [data[i : i + 8] for i in range(0, len(data), 8)]
    data = bytearray([int(i, 2) for i in data])
    with open("output.txt", "wb") as f:
        f.write(data)
    '''






input_data = wav_to_binary('a7r56tu_received.wav')

chirp, data_rate, samples = sweep()

known_ofdm_symbols = np.genfromtxt('a7r56tu_knownseq.csv', delimiter=',')

known_ofdm_symbols = [int(i) for i in known_ofdm_symbols]



PREFIXED_SYMBOL_LENGTH = len(known_ofdm_symbols)

recieve(input_data,chirp)
