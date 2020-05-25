"""
Modulation
One of:
    2PSK
    4PSK
    16QAM
"""
MODULATION = "2PSK"


"""
Encoding
One of:
    CONVOLUTION
"""
ENCODING = "CONVOLUTION"

SAMPLE_FREQUENCY = 44100

N = 1024 # IDFT length

PADDING = 0 # Frequency padding within block

CP = 32 # Length of cyclic prefix

BITS_PER_CONSTELLATION_VALUE = 2 # Length of binary word per constellation symbol

FILLER_VALUE = complex(0, 0) # Complex value to fill up partially full blocks

PILOT_FREQUENCY = 8 # Frequency of symbols to be pilot symbols

PILOT_SYMBOL = complex(1, 1) / np.sqrt(2) # Value of pilot symbol

