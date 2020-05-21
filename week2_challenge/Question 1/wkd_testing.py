import numpy as np
import scipy.signal as sg
from scipy.io import wavfile
from matplotlib import pyplot as plt

fig, axs = plt.subplots(4)
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

    #Normalise Data
    norm_data = [i/32767 for i in output_data]

    # Add 1 to make all values positive
    # Then scale by 2^16 / 2 = 2^15
    # Then convert to integer (rounds down)
    # Now we have 32 bit integers

    norm_data = [int((datum + 1) * np.power(2, 15)) for datum in norm_data]

    # Now convert to binary strings
    # Use zfill to make sure each string is 16 bits long
    # (By default python would not include redundant zeroes)
    # (And that makes it super hard to decode)
    # And use "".join() to make the whole thing one big string
    return "".join(format(datum, "b").zfill(16) for datum in norm_data)
 
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
    
def recieve(input_data, chirp):
   
    data = input_data
    data = np.array(data).astype(np.int16) 
    #data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()

    # Correlate
    prod = [datum * delayed_datum for datum, delayed_datum in zip(chirp,data)]
    axs[1].plot(prod)

    # Accumulate
    acc = [0]
    for datum in prod:
        acc.append(acc[-1] + datum)
    axs[2].plot(acc)

    # Differentiate
    diff = np.diff(acc)
    axs[3].plot(diff)

    # Extremify
    for i in range(len(diff)):
        if diff[i] >= 0:
            diff[i] = +1
        else:
            diff[i] = -1
    axs[3].plot(diff)

    # Detect symbols with a moving average window of width CP
    avg = []
    for i in range(len(diff[CP:])):
        avg.append(np.average(diff[i : i + CP]))
    axs[3].plot(avg)
    plt.show()

    chunks = [avg[i : i + PREFIXED_SYMBOL_LENGTH] for i in range(0, len(avg), PREFIXED_SYMBOL_LENGTH)]
    chunks[-1] += [0] * (PREFIXED_SYMBOL_LENGTH - len(chunks[-1]))
    scores = [0] * PREFIXED_SYMBOL_LENGTH
    threshold = 0.98
    for i in range(len(scores)):
        for chunk in chunks:
            if chunk[i] >= threshold:
                scores[i] += 1
    plt.plot(scores)
    plt.show()
    return 0 
    ###DO CHANNEL ESTIMATION HERE?##
    ## 1) Extract pilots from the signal as we know their positions
    ## 2) Average of each eqalised pilot over all OFDM symbols received
    ## 3) Interpolate over the data for channel estimation::
    ##      a) Could do interpolations between real and img data separately
    ##      b) Or could do interpolations between magnitude and phase separately
    ## 4) Each of the data carriers within each OFDM symbol is then equalised at its
    ##    corresponding frequency using the complex interpolated channel estimate.
    ##    Since the channel estimate is complex we can equalise both in magnitude and phase.
    """ Don't get step four """

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

chirp, data_rate, PREFIXED_SYMBOL_LENGTH = sweep()

known_ofdm_symbols = np.genfromtxt('a7r56tu_knownseq.csv', delimiter=',')

known_ofdm_symbols = [int(i) for i in known_ofdm_symbols]

recieve(input_data,chirp)

