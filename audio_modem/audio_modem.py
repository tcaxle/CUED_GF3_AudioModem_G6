"""
Structures
----------
(Lengths given for N=1024, CP=32, PADDING=0, CONSTELLATION=QPSK)

1. Binary Data:

    [< DATA >]
    |--1022--|

2. Paired Binary Data:

    [< DATA >]
    |-- 511--|

3. Blocks:

    [< 0 >|< PAD >|< DATA >|< PAD >|< 0 >|< PAD >|< CONJUGATE DATA >|< PAD >]
    |             |--0511--|                     |------- 511-------|       |
    |--------------------------------- 1024---------------------------------|

4. Symbols (after IFFT):

    [< SYMBOL >]
    |---1024---|

5. Prefixed Symbols:

    [< CYCLIC PREFIX >|< SYMBOL >]
    |------- 32-------|---1024---|
    |------------1056------------|

"""

"""
Imports
-------
"""
import numpy as np
import scipy as sp
from scipy import signal as sg
from scipy.io import wavfile
from matplotlib import pyplot as plt
import sounddevice as sd
import random

"""
Constants
---------
"""
# Set:
N = 1024 # IDFT length
PADDING = 0 # Frequency padding within block
CP = 32 # Length of cyclic prefix
BITS_PER_CONSTELLATION_VALUE = 2 # Length of binary word per constellation symbol
CONSTELLATION = {
    "00" : complex(+1, +1) / np.sqrt(2),
    "01" : complex(-1, +1) / np.sqrt(2),
    "11" : complex(-1, -1) / np.sqrt(2),
    "10" : complex(+1, -1) / np.sqrt(2),
} # Binary words mapped to complex values
SAMPLE_FREQUENCY = 44100 # Sampling rate of system
FILLER_VALUE = complex(0, 0) # Complex value to fill up partially full blocks
PILOT_FREQUENCY = 8 # Frequency of symbols to be pilot symbols
PILOT_SYMBOL = complex(1, 1) / np.sqrt(2) # Value of pilot symbol
# Calculated:
PREFIXED_SYMBOL_LENGTH = N + CP
CONSTELLATION_VALUES_PER_BLOCK = int((N - 2 - 4 * PADDING) / 2)
DATA_CONSTELLATION_VALUES_PER_BLOCK = CONSTELLATION_VALUES_PER_BLOCK - int(CONSTELLATION_VALUES_PER_BLOCK / PILOT_FREQUENCY)
DATA_BITS_PER_BLOCK = DATA_CONSTELLATION_VALUES_PER_BLOCK * BITS_PER_CONSTELLATION_VALUE

"""
sounddevice settings
--------------------
"""
sd.default.samplerate = SAMPLE_FREQUENCY
sd.default.channels = 1

def text_to_binary(input_file="input.txt"):
    """
    Parameters
    ----------
    input_file : FILE NAME (STRING)
        UTF-8 encoded text in a file

    Returns
    -------
    output_data : STRING
        string binary data
    """
    # Open the file and read the data
    with open(input_file, "rb") as f:
        output_data = f.read()

    # Encode text data as binary with utf-8 encoding
    # zfill ensures each word is 8 bits long
    return "".join(format(datum, "b").zfill(8) for datum in output_data)

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

    # Add 1 to make all values positive
    # Then scale by 2^16 / 2 = 2^15
    # Then convert to integer (rounds down)
    # Now we have 32 bit integers
    output_data = [int((datum + 1) * np.power(2, 15)) for datum in output_data]

    # Now convert to binary strings
    # Use zfill to make sure each string is 16 bits long
    # (By default python would not include redundant zeroes)
    # (And that makes it super hard to decode)
    # And use "".join() to make the whole thing one big string
    return "".join(format(datum, "b").zfill(16) for datum in output_data)

def fill_binary(input_data):
    """
    Makes sure that the length of the binary string will be an exact number of data blocks

    Parameters
    ----------
    input_data : STRING
        string of binary data
    Returns
    -------
    output_data : STRING
        string of binary data
    """
    output_data = [input_data[i : i + DATA_BITS_PER_BLOCK] for i in range(0, len(input_data), DATA_BITS_PER_BLOCK)]

    # Append 0s to data to make it correct length for integer number of constellation values and blocks
    output_data[-1] += "0" * (DATA_BITS_PER_BLOCK - len(output_data[-1]))

    return "".join(output_data)

def xor_binary_and_key(input_data):
    """
    XORs each bit with the key

    Parameters
    ----------
    input_data : STRING
        string of binary data
    Returns
    -------
    output_data : STRING
        string of binary data
    """
    def XOR(a, b):
        if a == b:
            return "0"
        else:
            return "1"

    # Open the file and read the data
    with open("key.txt", "r") as f:
        key = f.read()

    # make data into list of bits
    output_data = [datum for datum in input_data]

    # XOR the data
    for i in range(0, len(output_data), DATA_BITS_PER_BLOCK):
        for j in range(DATA_BITS_PER_BLOCK):
            output_data[i + j] = XOR(output_data[i + j], key[j])

    return "".join(output_data)

def binary_to_words(input_data):
    """
    Parameters
    ----------
    input_data : STRING
        string of binary data

    Returns
    -------
    output_data : LIST of [STRING]
        list of binary data words of length BITS_PER_CONSTELLATION_VALUE
    """
    # Split into word length blocks
    output_data = [input_data[i : i + BITS_PER_CONSTELLATION_VALUE] for i in range(0, len(input_data), BITS_PER_CONSTELLATION_VALUE)]

    if len(output_data[-1]) != BITS_PER_CONSTELLATION_VALUE:
        raise Exception("\n\nNot enough binary data to fill make words!\nFinal word of length {} out of {}!\n".format(len(output_data[-1]), BITS_PER_CONSTELLATION_VALUE))

    return output_data

def words_to_constellation_values(input_data):
    """
    Parameters
    ----------
    input_data : LIST of [STRING]
        list of binary words, length BITS_PER_CONSTELLATION_VALUE

    Returns
    -------
    output_data : LIST of COMPLEX
        list of complex valued data based on CONSTELLATION
    """
    output_data = []
    for word in input_data:
        if len(word) != BITS_PER_CONSTELLATION_VALUE:
            # Check word of correct length
            raise Exception("\n\nConstellation words must be of length {}!\n".format(BITS_PER_CONSTELLATION_VALUE))
        elif word not in CONSTELLATION.keys():
            # Check word in constellation
            raise Exception("\n\nInvalid constellation word {}!\n".format(word))
        else:
            # Append the complex value associated with that word
            output_data.append(CONSTELLATION[word])

    return output_data

def constellation_values_to_data_blocks(input_data):
    """
    Parameters
    ----------
    input_data : LIST of COMPLEX
        list of complex valued data

    Returns
    -------
    output_data : LIST of LIST of COMPLEX
        splits data into blocks of length CONSTELLATION_VALUES_PER_BLOCK and adds pilot symbols
    """

    # Split into blocks
    output_data = [input_data[i : i + DATA_CONSTELLATION_VALUES_PER_BLOCK] for i in range(0, len(input_data), DATA_CONSTELLATION_VALUES_PER_BLOCK)]
    # Add pilot symbols
    for i in range(len(output_data)):
        # split into smaller blocks
        block = output_data[i]
        output_data[i] = [output_data[i][j : j + PILOT_FREQUENCY - 1] + [PILOT_SYMBOL] for j in range(0, DATA_CONSTELLATION_VALUES_PER_BLOCK, PILOT_FREQUENCY - 1)]
        # Flatten and remove extra value
        output_data[i] = [datum for subblock in output_data[i] for datum in subblock][:511]

    return output_data

def conjugate_block(input_data):
    """
    Parameters
    ----------
    input_data : LIST of COMPLEX
        list of complex valued data

    Returns
    -------
    output_data : LIST of COMPLEX
        list of conjugates of input data
        NB: list is in reverse order of input data (mirrored)
    """

    # Find conjugates of reversed list
    return [np.conj(datum) for datum in input_data[::-1]]

def assemble_block(input_data):
    """
    Parameters
    ----------
    input_data : LIST of LIST of COMPLEX
        list of data blocks to be assembled ready to IDFT

    Returns
    -------
    output_data : LIST of LIST of COMPLEX
        list of blocks assembled ready for IDFT
    """
    padding = [0] * PADDING
    dc = [0]
    mid = [0]
    return [dc + padding + block + padding + mid + padding + conjugate_block(block) + padding for block in input_data]

def block_ifft(input_data):
    """
    Parameters
    ----------
    input_data : LIST of LIST of COMPLEX
        list of blocks to be transformed

    Returns
    -------
    output_data : LIST of LIST of FLOAT
        list of transformed blocks (now real valued)
    """
    return [np.fft.ifft(block, n=N).real.tolist() for block in input_data]

def cyclic_prefix(input_data):
    """
    Parameters
    ----------
    input_data : LIST of LIST of FLOAT
        list of transformed blocks (now real valued)

    Returns
    -------
    output_data : LIST of LIST of FLOAT
        list of transformed blocks with cyclic prefix
    """
    return [block[-CP:] + block for block in input_data]

def output(input_data, save_to_file=False, suppress_audio=False):
    """
    Parameters
    ----------
    input_data : LIST of LIST of FLOAT
        list of transformed blocks with cyclic prefix
    save_to_file : BOOL
        if set then outputs data "output.txt"
    suppress_audio : BOOL
        if set then does not output sound

    * Normalises data to +/- 1.0
    * Transmits data from audio device
    """
    # Pad with 0.1s of silence either side of transmitted data
    silent_padding = [0] * int(SAMPLE_FREQUENCY * 0.1)
    data = silent_padding + [datum for block in input_data for datum in block] + silent_padding
    # convert to 16-bit data
    data = np.array(data).astype(np.float32)
    # Normalise to 16-bit range
    data *= 32767 / np.max(np.abs(data))
    # start playback
    axs[0].plot(data)
    if not suppress_audio:
        sd.play(data)
        sd.wait()
    return data

def recieve(input_data):
    '''
    data = input_data
    delayed_data = [0] * N + input_data
    diff = [datum - delayed_datum for datum, delayed_datum in zip(data, delayed_data)]
    axs[1].plot(data)
    axs[2].plot(delayed_data)
    axs[3].plot(diff)
    plt.show()
    '''

    axs[0].plot(input_data)

    data = input_data
    data = np.array(data).astype(np.float32)
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()

    # Correlate
    delayed_data = [0] * N + data[:-N]
    prod = [datum * delayed_datum for datum, delayed_datum in zip(data, delayed_data)]
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
    diff = [1 if datum >= 0 else -1 for datum in diff]
    #axs[3].plot(diff)

    # Detect symbols with a moving average window of width CP
    avg = []
    for i in range(len(diff[CP:])):
        avg.append(np.average(diff[i : i + CP]))
    avg = [datum ** 3 for datum in avg]
    # Denoise
    avg = np.array(avg).astype(np.float32)
    avg *= 1.0 / np.max(np.abs(avg))
    avg = avg.tolist()
    axs[3].plot(avg)

    # Detect most common locations of cyclic prefix within the symbol
    chunks = [avg[i : i + PREFIXED_SYMBOL_LENGTH] for i in range(0, len(avg), PREFIXED_SYMBOL_LENGTH)]
    chunks[-1] += [0] * (PREFIXED_SYMBOL_LENGTH - len(chunks[-1]))
    scores = [0] * PREFIXED_SYMBOL_LENGTH
    threshold = 0.5
    for i in range(len(scores)):
        for chunk in chunks:
            if chunk[i] >= threshold:
                scores[i] += 1
    max_score= max(scores)
    shifts = []
    for i in range(len(scores)):
        if scores[i] == max_score:
            shifts.append(i)
    shifts = [shift + CP for shift in shifts]

    # Create windows
    # (for graphical reasons only)
    windows = [0] * len(data)
    for i in range(len(windows)):
        if i % PREFIXED_SYMBOL_LENGTH == 0:
            # Place marker for window starts
            windows[i] = 32768
            # Put marker for cyclic prefixes
            try:
                windows[i + CP] = 16384
            except:
                pass
            try:
                windows[i - CP] = 16384
            except:
                pass


    # For each possible shift value, retrioeve the first OFDM symbol
    deviations = {}
    for shift in shifts:
        # Shift data to synchronise
        shifted_data = data[shift:]
        shifted_data = [shifted_data[i : i + PREFIXED_SYMBOL_LENGTH] for i in range(0, len(shifted_data), PREFIXED_SYMBOL_LENGTH)]

        # Remove all data blocks whose power is less than the normalised cutoff power
        power_list = [np.sqrt(np.mean(np.square(block))) for block in shifted_data]
        power_list = np.array(power_list)
        power_list = power_list - np.min(power_list)
        power_list *= 1.0 / np.max(power_list)
        power_list = power_list.tolist()
        cutoff = 0.5
        power_list = [0 if datum < cutoff else 1 for datum in power_list]
        shifted_data = [shifted_data[i] for i in range(len(shifted_data)) if power_list[i] == 1]

        # Extract first symbol and remove cyclic prefix
        shifted_data = shifted_data[0][CP:]

        # FFT and extract encoded data
        shifted_data = np.fft.fft(shifted_data, n=N)
        shifted_data = shifted_data[1 : 1 + CONSTELLATION_VALUES_PER_BLOCK]

        # Check arguments of first quadrant
        # To check if it's a circle or a cluster
        shifted_data = [np.arctan(datum.imag / datum.real) for datum in shifted_data]
        shifted_data = [datum for datum in shifted_data if datum >= 0 and datum <= np.pi / 2]
        deviations[np.std(shifted_data)] = shift

    shift = deviations[min(deviations.keys())]

    # Plot windows shifted by shift
    windows = [0] * shift + windows[:-shift]
    axs[0].plot(windows)
    plt.show()

    # Shift data to synchronise
    data = data[shift:]
    data = [data[i : i + PREFIXED_SYMBOL_LENGTH] for i in range(0, len(data), PREFIXED_SYMBOL_LENGTH)]

    # Remove all data blocks whose power is less than the normalised cutoff power
    power_list = [np.sqrt(np.mean(np.square(block))) for block in data]
    power_list = np.array(power_list)
    power_list = power_list - np.min(power_list)
    power_list *= 1.0 / np.max(power_list)
    power_list = power_list.tolist()
    cutoff = 0.5
    power_list = [0 if datum < cutoff else 1 for datum in power_list]
    data = [data[i] for i in range(len(data)) if power_list[i] == 1]

    for block in data:
        block = block[CP:]
        block = np.fft.fft(block, n=N)
        block = block[1 : 1 + CONSTELLATION_VALUES_PER_BLOCK]
        plt.scatter(block.real, block.imag)
    plt.show()

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

def add_noise(input_data, SNR=1000):
    # Preprocess
    data = input_data
    data = np.array(data)
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()
    print(data[1])

    # Add AGWN
    SNR = (10) ** (SNR / 20)
    noise_magnitude = 1 / SNR
    noise = noise_magnitude * np.random.normal(0, 1, len(data))
    noise = noise.tolist()
    data = [datum + noise_datum for datum, noise_datum in zip(data, noise)]

    return data

def synchronise(input_data):
    input_data = np.array(input_data)

    L = int(N / 2)
    def calcP_R_M(rx_signal):
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

        P = sg.lfilter(b_P, a_P, mult) / L
        R = sg.lfilter(b_P, a_P, square) / L

        Pr = P[L:]
        Rr = R[L:]
        M = abs(Pr/Rr)**2
        return Pr, Rr, M  # throw away first L samples, as they are not correct due to filter causality

    Pr,Rr,M = calcP_R_M(input_data)   #Calculates preamble starting point by finding a plateu just before the preamble starts

    plt.plot(abs(Pr), 'b--', lw=3, label='$P(d)$ (equation (6))');
    plt.plot(abs(Rr), 'r--', lw=3, label='R, method 1')
    plt.legend()
    plt.show()

    plt.plot(abs(input_data), label='$r[n]$', color='cyan')
    plt.plot(M, label='$M(d)$')
    plt.xlim(0,2000)
    plt.legend()
    plt.show()


    plt.subplot(211)
    plt.plot(abs(input_data))

    plt.subplot(212)
    #Filter to turn plateau into an peak for detection
    b_toPeak = np.ones(L) / L
    a = (1,)
    M_filt = sg.lfilter(b_toPeak, a, M)


    plt.plot(M,label='M(d) Metric')
    #plt.plot(M_filt)

    #Differentiate the filtered data
    D = np.diff(M_filt)

    zeroCrossing_2 = ((D[:-1] * D[1:]) <= 0) * (M[1:-1] > 0.01)

    b_ignore = np.ones(1+1024+512)
    b_ignore[0] = 0
    ignore_times = (sg.lfilter(b_ignore, (1, ), zeroCrossing_2) > 0).astype(int)
    zeroCrossing_3 = zeroCrossing_2 * (ignore_times == 0)   # keep only the zero-crossings where the ignore-window is not on

    for i in range(len(zeroCrossing_3)):
        if zeroCrossing_3[i]:
            start = i
            break

    plt.plot(zeroCrossing_3, label='Preamble Start')
    plt.legend()
    plt.show()

    #We have used the first block of data as our preamble so the M metric finds a large match with both the preamble and the first block
    #Conditioning the signal to ignore matches less than the OFDM length means we only detect the preamble

    return start

def create_preamble():
    """
    Creates a preamble of length 2L = N
    """
    data = "".join([str(random.randint(0, 1)) for i in range(CONSTELLATION_VALUES_PER_BLOCK * BITS_PER_CONSTELLATION_VALUE)])
    data = binary_to_words(data)
    data = words_to_constellation_values(data)
    data = constellation_values_to_data_blocks(data)
    data = assemble_block(data)
    for i in range(len(data[0])):
        if i % 2 != 0:
            data[0][i] = 0
    data = block_ifft(data)
    data = cyclic_prefix(data)
    data = data[0]
    data = [2 * datum for datum in data]
    return data

def add_noise(input_data, amplitude):
    scale = max(input_data)
    noise = scale * amplitude * np.random.normal(0, 1, len(input_data))
    return [datum + noise_datum for datum, noise_datum in zip(input_data, noise)]

def transmit(input_file="input.txt", input_type="txt", save_to_file=False, suppress_audio=False):
    """
    Parameters
    ----------
    input_file : STRING
        name of the input file
    input_type : STRING
        "txt" for text input
        "wav" for wav input
    save_to_file : BOOL
        if set then outputs data "output.txt"
    suppress_audio : BOOL
        if set then does not output sound
    """

    data = text_to_binary()
    data = fill_binary(data)
    data = xor_binary_and_key(data)
    data = binary_to_words(data)
    data = words_to_constellation_values(data)
    data = constellation_values_to_data_blocks(data)
    data = assemble_block(data)
    data = block_ifft(data)
    data = cyclic_prefix(data)
    # preamble = create_preamble()
    # data = [preamble] + data
    # https://audio-modem.slack.com/archives/C013K2HGVL3
    data = output(data, suppress_audio=True)

    data = add_noise(data, 0.01)

    #data = add_noise(data)
    #plt.plot(data)
    #plt.show()

    #start = synchronise(data)
    #data = data[start:]
    #plt.plot(data)
    #plt.show()

    recieve(data)

fig, axs = plt.subplots(4)
transmit()

def generate_key():
    random_string = "".join([str(random.randint(0, 1)) for i in range(DATA_BITS_PER_BLOCK)])
    with open("key.txt", "w") as f:
        f.write(random_string)
