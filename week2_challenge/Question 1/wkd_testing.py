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
    #Normalise Data
    #norm_data = [i/32767 for i in output_data]

    # Add 1 to make all values positive
    # Then scale by 2^16 / 2 = 2^15
    # Then convert to integer (rounds down)
    # Now we have 32 bit integers

    #norm_data = [int((datum + 1) * np.power(2, 15)) for datum in norm_data]

    # Now convert to binary strings
    # Use zfill to make sure each string is 16 bits long
    # (By default python would not include redundant zeroes)
    # (And that makes it super hard to decode)
    # And use "".join() to make the whole thing one big string
    #return "".join(format(datum, "b").zfill(16) for datum in norm_data)
 
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
    f_sweep *= 32767 / np.max(np.abs(f_sweep))
    
    f_sweep = f_sweep.astype(np.int16)
    # Play noise
    # print(f_sweep)
    # recording = sd.playrec(f_sweep, sample_rate, channels=channels)
    # sd.wait()
    return f_sweep, sample_rate, samples
    
def recieve(input_data, known_data):
    data = input_data
    data = np.array(data).astype(np.float64)
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()
    
    dd_sample = np.gradient(np.gradient(known_data))
    dd_data = np.gradient(np.gradient(data))
    sample_rate = 48000
    #Correlation between sample and data, normalised
    corr = sg.correlate(dd_data, dd_sample, mode='full')
    
    #This normalised the corr, but it gives errors
    #corr = corr / np.sqrt(sg.correlate(dd_sample, dd_sample, mode='full') * sg.correlate(dd_data, dd_data, mode='full'))
    
    #Create and shift x axis from -0.5 to 0.5
    #delay_arr = np.linspace(-0.5*len(known_data)/sample_rate,0.5*len(known_data)/sample_rate,len(known_data))
    
    #Estimates the point at which the peak correlation occurs  //This is not robust enough, needs smarter method
    lag_sample = np.argmax(corr)
    print(lag_sample)
    print(len(data))
    
    lag = []
    for i,datum in enumerate(data):
        if i == lag_sample:
            lag = i
    
    if lag < 0:
        print('data is ' + str(np.round(abs(lag),3)) + 's ahead of the sample')
    else:
        print('data is ' + str(np.round(lag,3)) + ' behind the sample')

 
    plt.figure()
    plt.plot(corr)
    plt.plot(lag)
    plt.title('Lag: ' + str(lag/sample_rate) + ' s or ' + str(lag) + ' samples')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.xlim(0,lag+10000)
    plt.show()
    
    
    data = data[lag+sample_rate:] #remove everything up to the end of the chirp
    
    plt.plot(data)
    plt.xlim(0,120000)
    print(data[sample_rate])
    #return lag, dd_data, dd_sample 
    N = 4096
    CYCLIC_PREFIX = 0 
    block_length = N + CYCLIC_PREFIX
    block_number = int(len(data) / block_length)
    
    # 1) Split into blocks of 4096
    data = np.array_split(data, block_number)
    
    # 3) DFT N = 4096
    demodulated_data = np.fft.fft(data, N)
    
    # # 4) Convolve with inverse FIR channel
    # # 4.1) Invert the channel
    # inverse_channel = np.fft.fft(channel, N)
    # # 4.2) Convolve
    # unconvolved_data = [np.divide(block, inverse_channel) for block in demodulated_data]
    # # 4.3) Discard last half of each block
    # unconvolved_data = [block[1:512] for block in unconvolved_data]
    
    # # 5) Decide on optimal decoder for constellations
    # # 5.1) Define constellation
    # constellation = {
    #     complex(+1, +1): (0, 0),
    #     complex(-1, +1): (0, 1),
    #     complex(-1, -1): (1, 1),
    #     complex(+1, -1): (1, 0),
    # }
    # # 5.2) Minimum distance decode and map to bits
    # mapped_data = []
    # for block in unconvolved_data:
    #     minimum_distance_block = []
    #     for data_symbol in block:
    #         # Get distance to all symbols in constellation
    #         distances = {abs(data_symbol - constellation_symbol): constellation_symbol for constellation_symbol in constellation.keys()}
    #         # Get minimum distance
    #         minimum_distance = min(distances.keys())
    #         # Find symbol matching minimum distance
    #         symbol = distances[minimum_distance]
    #         # Append symbol data to mapped data array
    #         mapped_data.append(constellation[symbol][0])
    #         mapped_data.append(constellation[symbol][1])

    
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
