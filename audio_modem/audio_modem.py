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

def text_to_binary(input_file="input.txt"):
    """
    Parameters
    ----------
    input_file : FILE NAME (STRING)
        UTF-8 encoded text in a file

    Returns
    -------
    output_data : LIST of [STRING]
        list of binary data
    """
    pass

def wav_to_binary(input_file="input.wav"):
    """
    Parameters
    ----------
    input_file : FILE NAME (STRING)
        wave data stored as floats in a .wav file

    Returns
    -------
    output_data : LIST of [STRING]
        list of binary data where each item is a "1" or "0"
    """
    pass

def word_length_binary(input_data):
    """
    Parameters
    ----------
    input_data : LIST of [STRING]
        list of binary data where each item is a "1" or "0"

    Returns
    -------
    output_daata : LIST of [STRING]
        list of binary data made up to an integer multiple of WORD_LENGTH
    """
    pass

def binary_to_words(input_data):
    """
    Parameters
    ----------
    input_data : LIST of [STRING]
        list of binary data where each item is a "1" or "0"

    Returns
    -------
    output_data : LIST of [STRING]
        list of binary words
        each word has length WORD_LENGTH
    """
    pass

def pairs_to_constellation_values(input_data):
    """
    Parameters
    ----------
    input_data : LIST of [STRING]
        list of binary words

    Returns
    -------
    output_data : LIST of COMPLEX
        list of complex valued data based on CONSTELLATION
    """
    pass

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
    pass

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
    pass

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
    pass

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
    pass

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
    pass

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
    pass

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

