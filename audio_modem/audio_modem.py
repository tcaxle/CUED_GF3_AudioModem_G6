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
WORD_LENGTH = 2 # Length of binary word per constellation symbol
CONSTELLATION = {
    "00" : complex(+1, +1) / np.sqrt(2),
    "01" : complex(-1, +1) / np.sqrt(2),
    "11" : complex(-1, -1) / np.sqrt(2),
    "10" : complex(+1, -1) / np.sqrt(2),
} # Binary words mapped to complex values
SAMPLE_FREQUENCY = 44100 # Sampling rate of system
FILLER_VALUE = complex(0, 0) # Complex value to fill up partially full blocks
# Calculated:
DATA_BLOCK_LENGTH = int((N - 2 - 4 * PADDING) / 2)
PREFIXED_SYMBOL_LENGTH = N + CP

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

def binary_to_words(input_data):
    """
    Parameters
    ----------
    input_data : STRING
        string of binary data

    Returns
    -------
    output_data : LIST of [STRING]
        list of binary data words of length WORD_LENGTH
    """
    # Split into word length blocks
    output_data = [input_data[i : i + WORD_LENGTH] for i in range(0, len(input_data), WORD_LENGTH)]

    # Make up final word to WORD_LENGTH with 0s
    output_data[-1] += "0" * (WORD_LENGTH - len(output_data[-1]))

    return output_data

def words_to_constellation_values(input_data):
    """
    Parameters
    ----------
    input_data : LIST of [STRING]
        list of binary words, length WORD_LENGTH

    Returns
    -------
    output_data : LIST of COMPLEX
        list of complex valued data based on CONSTELLATION
    """
    output_data = []
    for word in input_data:
        if len(word) != WORD_LENGTH:
            # Check word of correct length
            raise Exception("Constellation words must be of length {}!".format(WORD_LENGTH))
        elif word not in CONSTELLATION.keys():
            # Check word in constellation
            raise Exception("Invalid constellation word {}!".format(word))
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
        splits data into blocks of length DATA_BLOCK_LENGTH
        Makes up final block to full length with FILLER_VALUE
    """
    # Split data into blocks, length = DATA_BLOCK_LENGTH
    output_data = [input_data[i : i + DATA_BLOCK_LENGTH] for i in range(0, len(input_data), DATA_BLOCK_LENGTH)]

    # Add filler to final block
    output_data[-1] += [FILLER_VALUE] * (DATA_BLOCK_LENGTH - len(output_data[-1]))

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
    # Find conjugates
    output_data = [np.conj(datum) for datum in input_data]

    # Reverse the list
    return output_data[::-1]

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
    #axs[0].plot(data)
    sd.play(data)
    sd.wait()
    return data

def tomsrecieve(input_data):

    '''
    data = input_data
    delayed_data = [0] * N + input_data
    diff = [datum - delayed_datum for datum, delayed_datum in zip(data, delayed_data)]
    axs[1].plot(data)
    axs[2].plot(delayed_data)
    axs[3].plot(diff)
    plt.show()
    '''

    # Preprocess
    data = input_data
    data = np.array(data)
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()
    print(data[1])

    # Add AGWN
    SNR = 20 # dB
    SNR = (10) ** (SNR / 20)
    noise_magnitude = 1 / SNR
    noise = noise_magnitude * np.random.normal(0, 1, len(data))
    noise = noise.tolist()
    data = [datum + noise_datum for datum, noise_datum in zip(data, noise)]

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
    for i in range(len(diff)):
        if diff[i] >= 0:
            diff[i] = +1
        else:
            diff[i] = -1
    #axs[3].plot(diff)

    # Detect symbols with a moving average window of width CP
    avg = []
    for i in range(len(diff[CP:])):
        avg.append(np.average(diff[i : i + CP]))

    axs[3].plot(avg)
    plt.show()

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
    data = "".join([str(random.randint(0, 1)) for i in range(DATA_BLOCK_LENGTH * WORD_LENGTH)])
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

    #data = text_to_binary()
    random_string = "".join([str(random.randint(0, 1)) for i in range(10000)])
    data = random_string
    data = binary_to_words(data)
    data = words_to_constellation_values(data)
    data = constellation_values_to_data_blocks(data)
    data = assemble_block(data)
    data = block_ifft(data)
    data = cyclic_prefix(data)
    preamble = create_preamble()
    data = [preamble] + data

    data = output(data)

    #data = add_noise(data)
    #plt.plot(data)
    #plt.show()

    start = synchronise(data)
    data = data[start:]
    plt.plot(data)
    plt.show()

    #recieve(data)

#fig, axs = plt.subplots(4)
transmit()

