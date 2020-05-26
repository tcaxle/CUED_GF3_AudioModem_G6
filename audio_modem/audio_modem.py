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
    if save_to_file:
          # Write data
        with open('output.txt', 'w') as f:
            for i in data:
                f.write(str(i) + ',')
    return data

def recieve(input_data):
    
    ##### NOTES FOR THE FUTURE: 
    ##### 1) THIS FILE IS DIFFICULT TO PARSE DUE TO POOR FORMATTING (APOLOGIES)
    ##### 2) SYNCHRONISATION ONLY WORKS IF DATA DRIFTS FORWARD
    ##### 3) IF THE DATA DRIFTS BACKWARDS, ESTIMATES GIVE 0 DELAY,
    ##### SYNCHRO FUNCTION SHIFTS NOTHING
    ##### 4) CORRELATION DOESN'T WORK IN THAT CASE, BOTH MAX OR GRADIENT MAX FAIL
    ##### 5) FOR  DELAYS BOTH IN THE BEGGINING AND END OF FILES, 
    ##### ITERATION THROUGH THE shift AND SYNCHRO FUNCTIONS SHOULD: CLEAR THEM ALL,
    ##### THEN ALIGN THE DATA WITH THE SAMPLE AND ADD ZEROES EVERYWHERE ELSE. 
    ##### THIS WILL BE POSSIBLE IF BACKWARDS DRIFT IS HANDLED, NOT YET IMPLEMENTED
    ##### 6) THE PLOTTING FUNCTION IS FOR TESTING ONLY, NOT REQUIRED
    
    def shift_finder(sample, data, sample_rate, plot=False, grad_mode = True):
        
        """
        Takes a file to be sent and a received file and finds the difference 
        by which the received file is delayed.
        
        If plot is set, then it will produce a matplotlib plot of the output
        
        Grad Mode: True or False.
        
        If true, it finds the double gradient before correlating. 
        
        If False it just finds correleation between given inputs.
        
        Gradient estimation has yet to be tested fully.
            
        """
    
        
       
        dd_sample = sample
        dd_data = data
     
        if grad_mode: 
        ###Using the second derivative of signals
            dd_sample = np.gradient(np.gradient(sample))
            dd_data = np.gradient(np.gradient(data))
     
        #Correlation between sample and data, normalised
        corr = sg.correlate(dd_data, dd_sample, mode='full')
        
        #This normalised the corr, but it gives errors
        #corr = corr / np.sqrt(signal.correlate(dd_sample, dd_sample, mode='')[int(n/2)] * signal.correlate(dd_data, dd_data, mode='same')[int(n/2)])
        
        #Create and shift x axis from -0.5 to 0.5
        #delay_arr = np.linspace(-0.5*n/sample_rate, 0.5*n/sample_rate, n)
        
        #Estimates the point at which the peak correlation occurs  //This is not robust enough, needs smarter method
        shift = np.argmax(corr)
        
        if shift < 0:
            print('data is ' + str(np.round(abs(shift),3)) + 's ahead of the sample')
        else:
            print('data is ' + str(np.round(shift,3)) + ' behind the sample')
    
        
            shifts = np.linspace(shift-50,shift+50,101).astype(int).tolist()
            
            return shifts
     
# def shift_sync(sample, data, sample_freq, shift):
     
#     """
#     This function takes two data sets, their sampled frequency,
#     and the shift between them. 
    
#     It removes the shift between the beggining of the data (relative 
#     to the sample). 
    
#     It returns the data with equal length as the sample (for plotting).
#     """
    
#     ## shift should be more precise than sample rate
#     # Round order of magnitude of shift close to sample precision and add 1 for safety  
    
#     shift_sign = shift
#     #print(shift,type(shift))
#     shift = abs(np.round(shift,int(abs(np.floor(np.log10(abs(shift)))))+1))
    
#     # Find the difference between the two samples
#     sample_shift = int(np.floor(shift * sample_freq))
    
#     #Assuming the data has a delay at the beginning
    
#     if shift_sign > 0:
                
#         #remove the sample shift from the data, getting closer to when the sample began
#         data = data[sample_shift:]
        
#         #Pad data with sample_shift amount of zeroes so the lengths of the arrays match
#         data = np.concatenate((data,(np.zeros(sample_shift,dtype=np.int16))),axis=None)
              
        
#     ###Assuming the data arrives faster than the sample (negative delay)
#     ###This occurs if we estimate the shift too early. Feeding the signal again should
#     ###trigger this and try and shift the data to the right
    
        
#     elif shift_sign < 0:
        
#         #Pad sample_shift amount of zeroes until data and sample match    
#         data = np.concatenate(((np.zeros(sample_shift,dtype=np.int16)), data),axis=None)

#         #remove the end samples
#         data = data[:-sample_shift]
        
#     return data

def check_synchronisation(data,shifts):  
    
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

    return deviations[min(deviations.keys())]


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

def synchronise(input_data,CP):
    input_data = np.array(input_data)

    L = int(N / 2)
    
    def schmidl_cox(data,L):
        
        
        """
        P-metric:
        
        If the conjugate of a sample from the first half (r*_d) is multiplied 
        by the corresponding sample from the second half (r_d+L), the effect of
        the channel should cancel. Therefore, the products of these pairs will be
        very large.
        
        This is an iterative method as described in S&C to create a list of P-metric
        values for the entire received data.
        
        Notes: Misses the last 2L points of data, unsure if this might lead to error
                Not calculating P0 and R0, to save time, assumed irrelevant
        """
    
        P = [None]*(len(data))
        R = [None]*(len(data))
        M = [None]*len(data)
        
        P[0] = 0
        R[0] = 0
        
        for d in range(len(P)-2*L):        
            P[d+1] = P[d] + np.conj(data[d+L])*data[d+2*L] - np.conj(data[d])*data[d+L]
            
        """
        R-metric:
            Received energy of data. Operation for item d:
                --> add all squared values of items between d+L and d+2L
        """
        for d in range(len(R)-2*L):
            R[d+1] = R[d] + abs(data[d+2*L])**2 - abs(data[d+L])**2
        
        for d in range(len(M)-2*L):
            if R[d] != 0:
              M[d] = (abs(P[d])**2)/(R[d]**2) 
            else:
                M[d] = 0
                
        # plt.subplot(211)   
        # plt.plot(P,'b',label="P Metric")
        # plt.plot(R,'r',label="R Metric")
        # plt.subplot(212)
        # plt.plot(M,'y',label="M metric")
        # plt.legend()
        # plt.show()
        
        ##### Remove None values here
        P = [datum for datum in P if datum != None]
        
        R = [datum for datum in R if datum != None]
        
        M = [datum for datum in M if datum != None]
        
        return np.array(P), np.array(R), np.array(M)
        
    
    def synchro_samples(P,R,M,CP,N):
    
        # Low Pass Filter to smooth out plateau and noise
        num = np.ones(CP)/CP
        
        den = (1,0)
        
        Mf = sg.lfilter(num, den, M)
         
        # plt.subplot(212)    
        # plt.plot(M,label='M Metric')
        # plt.plot(Mf,'r',label = 'Filtered M Metric')
        # plt.show()
        
        #Differentiation turn peaks from the filtered metric into zero crossings
        
        Mdiff = np.diff(Mf)
        
        plt.plot(Mdiff,'r',label = 'Diff Mf Metric')
        plt.xlim(4230,5780)
        plt.show()
    
    
        ##Finds All zero crossings that match an M value above a threshold to account for noise
        # Threshold is 0.98, with noise it should be smaller
        
        zero_crossings = ((Mdiff[1:] * Mdiff[:-1])<=0)
       
        zero_crossings = zero_crossings*(M[1:-1]>0.98)
      
        ##Multple crossings due to noise. To avoid, after the first crossing we change the next 
        # N+CP crossings into False. 
        Len = len(zero_crossings)-N-CP-1
        
        "NEEDS CLEAN UP"
        for i in range(Len):
            if zero_crossings[i] == True:
                for j in range(i+1,N+CP+i+1):
                    zero_crossings[j] = False
        
        start =  [i for i, val in enumerate(zero_crossings) if val] 
        
        ##Only take the first detection, but we should be testing all of them
        
        start=start[0]
        
        return start
    
    
    P, R, M = schmidl_cox(input_data, L)
    start = synchro_samples(P,R,M,CP,N)
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
    #preamble = create_preamble()
    #data = [preamble] + data
    # https://audio-modem.slack.com/archives/C013K2HGVL3
    data = output(data,save_to_file=True, suppress_audio=True)

    data = add_noise(data, 0.05)

    plt.plot(data)
    plt.show()

    #start = synchronise(data,CP)
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
