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