"""
Structures
----------
(Lengths given for N=4096, CP=704 (B), PADDING=0, CONSTELLATION=QPSK)

MODE A) CP = 224, MODE B) CP = 704, MODE C) CP = 1184 

1. Binary Data:

    [< DATA >]
    |--4094--|

2. Paired Binary Data:

    [< DATA >]
    |--2047--|
_
3. Blocks:

    [< 0 >|< DATA >|< 0 >|< CONJUGATE DATA >|]
    |     |--2047--|     |------- 2047-------|
    |---------------4096---------------------|

4. Symbols (after IFFT):

    [< SYMBOL >]
    |---4096---|

5. Prefixed Symbols:

    [< CYCLIC PREFIX >|< SYMBOL >]
    |-------704-------|---4096---|
    |------------4800------------|

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
from data_handling.exports import wav_output

"""
Constants
---------
"""
# Set:
N = 4096 # DFT length
PADDING = 0 # Frequency padding within block
CP = 704 # Length of cyclic prefix

BITS_PER_CONSTELLATION_VALUE = 2 # Length of binary word per constellation symbol

CONSTELLATION = {
    "00" : complex(+1, +1) / np.sqrt(2),
    "01" : complex(-1, +1) / np.sqrt(2),
    "11" : complex(-1, -1) / np.sqrt(2),
    "10" : complex(+1, -1) / np.sqrt(2),
} # Binary words mapped to complex values

SAMPLE_FREQUENCY = 48000 # Sampling rate of system

FILLER_VALUE = complex(0, 0) # Complex value to fill up partially full blocks

# Calculated:
PREFIXED_SYMBOL_LENGTH = N + CP #4800
CONSTELLATION_VALUES_PER_BLOCK = int((N - 2 - 4 * PADDING) / 2) #2047
DATA_BITS_PER_BLOCK = CONSTELLATION_VALUES_PER_BLOCK * BITS_PER_CONSTELLATION_VALUE #4094
BLOCK_LENGTH = 180
CHIRP_LENGTH = 5
KNOWN_BLOCK_LENGTH = 20
FRAME_LENGTH = (CHIRP_LENGTH + KNOWN_BLOCK_LENGTH + BLOCK_LENGTH + KNOWN_BLOCK_LENGTH)*PREFIXED_SYMBOL_LENGTH

"""
sounddevice settings
--------------------
"""
sd.default.samplerate = SAMPLE_FREQUENCY
sd.default.channels = 1

def check_typing(input_data):
    def recur(input_data):
    
        try:
            print("\nType: {}\nLength: {}".format(type(input_data), len(input_data)))
        except:
            print("\nType: {}".format(type(input_data)))
    
        if type(input_data) == list:
            type_zero = type(input_data[0])
            for item in input_data:
                if type(item) != type_zero:
                    raise Exception("Inconsistent List!")
            if type_zero == list:
                recur(input_data[0])
            else:
                print("\nType: {}".format(type_zero))
    
    recur(input_data)

def norm(input_data): 
    input_data = np.array(input_data).astype(np.float32)
    input_data *= 1/np.max(np.abs(input_data))
    return input_data.tolist()   

#######TRANSMITTER########


def sweep(f_start=2000, f_end=4000, sample_rate=SAMPLE_FREQUENCY,samples=5*(N+CP), channels=1):
    """
    Returns a frequency sweep
    """
    # Calculate the duration
    duration = samples / sample_rate
    # Produce time array
    time_array = np.linspace(0, duration, samples)
    
    # Produce frequency sweep
    f_sweep = sg.chirp(time_array, f_start, duration, f_end)
    # Normalise sweep
    # f_sweep *= 1/np.max(np.abs(f_sweep))
    # f_sweep = f_sweep.astype(np.int16)
    
    return f_sweep

def get_known_data(save=False):
    with open("random_bits.txt", "r") as f:
        data = f.read()
        
    data = data[: DATA_BITS_PER_BLOCK]
    # We need the equivalent of 1 OFDM symbol length and 
    # We can repeat it 20 times as needed
    # Where would it be best to cut-off uneccesary data? In the file itself?

    data = binary_to_words(data)
    data = words_to_constellation_values(data)
    data = constellation_values_to_data_blocks(data)
    data = assemble_block(data)
    data = block_ifft(data)
    data = cyclic_prefix(data)    
    if save:
        with open('known_data.txt','w') as f:
            for i in range(len(data[0])):
                f.write(str(data[0][i]) + ',')

    #Don't quite get why we need the first element specifically. Is this due to 
    #the cyclic prefix operation?

    return data[0]    

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
    with open("random_bits.txt", "r") as f:
        key = f.read()
        
    key = key[:N]
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
        splits data into blocks of length CONSTELLATION_VALUES_PER_BLOCK
    """
        
    # Split into blocks
    output_data = [input_data[i : i + CONSTELLATION_VALUES_PER_BLOCK] for i in range(0, len(input_data), CONSTELLATION_VALUES_PER_BLOCK)]
    
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
    if not type(input_data) == list:
        input_data = input_data.tolist()

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

def assemble_frame(input_data):
    #input_data = List of List of Float
    #known_data = List of List of Float
    #Deal with known data
    input_data = norm(input_data)
    
    #We have 7 blocks so 7P floats
    #We want to make 1 frame with the 7P floats + 173P zeroes
    
    #we have x blocks of ofdm symbols
    #we need ot divide by 180 so we get x/180 frames +1 frame if there's remainder
    
    no_frames = int(np.ceil(len(input_data)/BLOCK_LENGTH))
    
    #check_typing(input_data)
    
    frames = [None]*no_frames
    #We are now getting 7 frames with P floats in each frame, using only the first P frame
    
    chirp = sweep(f_start=2000,f_end=4000).tolist() #Length 5(N+CP)
    

    known_data = norm(20*get_known_data()) #Length 20(N+CP)

    #check_typing(input_data)
    #List of List of List of Float
    output_data = [input_data[i : i + BLOCK_LENGTH] for i in range(0, len(input_data), BLOCK_LENGTH)]
   
    #List of List of Float
    for i,frame in enumerate(output_data):
        output_data[i]  = [item for sublist in frame for item in sublist]
   
   # zero_block =[0]*PREFIXED_SYMBOL_LENGTH

     
    #Needs to be appended with zeroes
    output_data[-1] += [0.0] * (PREFIXED_SYMBOL_LENGTH*BLOCK_LENGTH - len(output_data[-1]))
    # for i in range(BLOCK_LENGTH-len(output_data)):
    #     output_data.append(zero_block)
    
    #check_typing(output_data)
    
    #output_data = [floats for frames in output_data for floats in blocks]
    #check_typing(output_data)
    
    frames = [chirp + known_data + block + known_data for block in output_data]
   
    #check_typing(frames)     
    #Flatten frames
    
    
    return frames

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
    data = input_data
    
    for i,block in enumerate(data):
            data[i] = np.array(block).astype(np.float32)
            # Normalise to 16-bit range
            data[i] *= 32767 / np.max(np.abs(block))
            data[i] = data[i].tolist()

    # Pad with 0.1s of silence either side of transmitted data
    silent_padding = [0] * int(SAMPLE_FREQUENCY * 0.1)
    data = silent_padding + [datum for block in data for datum in block] + silent_padding
    
    # # convert to 16-bit data
    # data = np.array(data).astype(np.float32)
    # # Normalise to 16-bit range
    # data *= 32767 / np.max(np.abs(data))
    # # start playback
    #axs[0].plot(data)
    
    if not suppress_audio:
        sd.play(data)
        sd.wait()
    
    if save_to_file:
          # Write data
        with open('output.txt', 'w') as f:
            for i in data:
                f.write(str(i) + ',')
    
    return data

def transmit(input_file="input.txt", input_type="txt", save_to_file=False, suppress_audio=False,DEBUG=False):
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
    if DEBUG:    
        data = text_to_binary()
        data = fill_binary(data)
        data = xor_binary_and_key(data)
        print(data[:10])
        print("Length of data:", len(data))
        data = binary_to_words(data)
        print("")
        print(data[:10])
        print("Length of words:", len(data))
        data = words_to_constellation_values(data)
        print("")
        print(data[:10])
        print("Length of constellation:", len(data))
        data = constellation_values_to_data_blocks(data)
        print("")
        print(data[0][:10])
        print("Number of data blocks:", len(data))
        print("Length of data blocks:", len(data[0]))
        #data= data[:4096]
        print(type(data))
        data = assemble_block(data)
        print("")
        print(data[0][:10])
        print("Number of assembled blocks:", len(data))
        print("Length of assembled blocks:", len(data[0]))
        
        data = block_ifft(data)
        print("")
        print(data[0][:10])
        print("Number of assembled blocks:", len(data))
        print("Length of IFFT:", len(data[0]))
        data = cyclic_prefix(data)
        print("")
        print(data[0][:10])
        print("Number of CPed blocks:", len(data))
        print("Length of CPed blocks:", len(data[0]))
        
        
        chirp = sweep(f_start=2000,f_end=4000)
        
        data = assemble_frame(data)
        
        print("\n",type(data),"\n",type(data[0]))
        print("\n",len(data),"\n",len(data[0]))
        return data
        
        # data = output(data,save_to_file=False,suppress_audio=True)
        # print("")
        # print("Padding adds 1 block before and 1 block after")
        # print("Number of output:", len(data))
        # print(data[15000:15430])
        
        # wav_output(data,SAMPLE_FREQUENCY)
        # wav_output(chirp,SAMPLE_FREQUENCY)
    
    data = text_to_binary()
    data = fill_binary(data)
    data = xor_binary_and_key(data)
    data = binary_to_words(data)
    data = words_to_constellation_values(data)
    data = constellation_values_to_data_blocks(data)
    data = assemble_block(data)
    data = block_ifft(data)
    data = cyclic_prefix(data)
    data = assemble_frame(data)
    data = output(data,save_to_file=False, suppress_audio=True)

    return data
    ###Schmidl and Cox, Deprecated
    # #preamble = create_preamble()
    # #data = [preamble] + data
    #data = add_noise_db(data, 3)
    #start = synchronise(data,CP)
    #data = data[start:]
    #plt.plot(data)
    #plt.show()
    #return data

#fig, axs = plt.subplots(1)


#######RECEIVER########
def shift_finder(sample, data, sample_rate, window=50, plot=False, grad_mode = True):
    """
    Takes a file to be sent (chirp) and a received file and tries to locate
    the chirp inside the received file
    
    If plot is set, then it will produce a matplotlib plot of the output
    
    Grad Mode: True or False.
    
    If true, it finds the second gradient before correlating. 
    
    If False it just finds correleation between given inputs.
    
    window: INT How far before and afterwards to search for synchronisation
    """
    
    if window > len(sample):
        raise ValueError("The window should not be larger than the added chirp")
    
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
        print('data is ' + str(np.round(abs(shift),3)) + 's ahead of the sample, something is wrong')
    else:
        print('data is ' + str(np.round(shift,3)) + ' behind the sample')
    
    shifts = np.linspace(shift-window,shift+window,2*window+1).astype(int).tolist()
        
    return shifts

def shift_sync(sample, data, sample_freq, shift):
    """
    This function takes two data sets, their sampled frequency,
    and the shift between them. 

    It removes the shift between the beggining of the data (relative 
    to the sample). 

    It returns the data with equal length as the sample (for plotting).
    """

    ## shift should be more precise than sample rate
    # Round order of magnitude of shift close to sample precision and add 1 for safety  

    shift_sign = shift
    #print(shift,type(shift))
    shift = abs(np.round(shift,int(abs(np.floor(np.log10(abs(shift)))))+1))

    # Find the difference between the two samples
    sample_shift = int(np.floor(shift * sample_freq))

    #Assuming the data has a delay at the beginning

    if shift_sign > 0:
        #remove the sample shift from the data, getting closer to when the sample began
        data = data[sample_shift:]
      
        #Pad data with sample_shift amount of zeroes so the lengths of the arrays match
        data = np.concatenate((data,(np.zeros(sample_shift,dtype=np.int16))),axis=None)
      
    ###Assuming the data arrives faster than the sample (negative delay)
    ###This occurs if we estimate the shift too early. Feeding the signal again should
    ###trigger this and try and shift the data to the right
      
    elif shift_sign < 0:
        #Pad sample_shift amount of zeroes until data and sample match    
        data = np.concatenate(((np.zeros(sample_shift,dtype=np.int16)), data),axis=None)

        #remove the end samples
        data = data[:-sample_shift]
      
    return data

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

def add_noise_db(input_data, SNR=1000):
    # Preprocess
    data = input_data
    data = np.array(data)
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()

    # Add AGWN
    SNR = (10) ** (SNR / 20)
    noise_magnitude = 1 / SNR
    noise = noise_magnitude * np.random.normal(0, 1, len(data))
    noise = noise.tolist()
    data = [datum + noise_datum for datum, noise_datum in zip(data, noise)]

    return data

def add_noise_amp(input_data, amplitude):
    scale = max(input_data)
    noise = scale * amplitude * np.random.normal(0, 1, len(input_data))
    return [datum + noise_datum for datum, noise_datum in zip(input_data, noise)]

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
    
            
        P = [0]*(len(data))
        R = [0]*(len(data))
        M = [0]*len(data)
        
        s=0
        for m in range(L):
           s +=  np.conj(data[m])*data[m+L]
        
        P[0]= s
        
        s=0
        for m in range(L):
            s += (data[m+L])**2
    
        R[0] = s
    
        ##Non causal version for the first 2L elements
        for d in range(2*L):        
            P[d+1] = P[d] + np.conj(data[d+L])*data[d+2*L] - np.conj(data[d])*data[d+L]
        ##Causal version for the rest of the data
        for d in range(2*L,len(P)-1):
              P[d+1] = P[d] + np.conj(data[d-L])*data[d] - np.conj(data[d-2*L])*data[d-L]      
            
        """
        R-metric:
            Received energy of data. Operation for item d:
                --> add all squared values of items between d+L and d+2L
        """
        ##Non causal version for the first 2L elements
        for d in range(2*L):
            R[d+1] = R[d] + abs(data[d+2*L])**2 - abs(data[d+L])**2
        
        ##Causal version for the rest of the data
        for d in range(L,len(R)-1):
            R[d+1] = R[d] + abs(data[d])**2 - abs(data[d-L])**2
    
        
        """
        M-metric: P squared over R squared
    
        """
        ##Set a threshold for the minimum R used, to reduce wrong detections
        R = np.array(R)    
        energy_threshold = np.sqrt(np.mean(R**2))
        
        
        for d in range(len(M)):
            if R[d] > (energy_threshold):
             M[d] = (abs(P[d])**2)/(R[d]**2) 
                    
        #plt.subplot(211)   
        #plt.plot(P,'b',label="P Metric")
        #plt.plot(R,'r',label="R Metric")
        # plt.subplot(212)
        # plt.plot(M,'y',label="M metric")
        # plt.legend()
        # plt.show()
    
        return np.array(P), np.array(R), np.array(M)
        
    
    def snc_start(P,R,M,ofdm_block_length,cp,threshold=0.9):

        # Low Pass Filter to smooth out plateau and noise
        num = np.ones(cp)/cp
        
        den = (1,0)
        
        Mf = sg.lfilter(num, den, M)
         
        #Differentiation turn peaks from the filtered metric into zero crossings
        
        Mdiff = np.diff(Mf)
        
    
        ##Finds All zero crossings that match an M value above a threshold to account for noise
        # Threshold is 0.98, with noise it should be smaller
        
        zero_crossings = ((Mdiff[:-1] * Mdiff[1:])<=0)*(M[1:-1]>threshold)
       
        ##Multple crossings due to noise. To avoid, after the first crossing we skip the next 
        # N+CP crossings. 
        
        ignored_crossings = np.ones(1+ofdm_block_length+cp) 
        ignored_crossings[0] = 0  
        ignore_times = (sg.lfilter(ignored_crossings, (1, ), zero_crossings) > 0).astype(int)
        zero_crossings = zero_crossings * (ignore_times == 0)   
            
    
        return  [i for i, val in enumerate(zero_crossings) if val] 

    

    P, R, M = schmidl_cox(input_data, L)
    start = snc_start(P,R,M,N,CP)
    shift = check_synchronisation(input_data, start)
    
    return shift

def generate_key():
    random_string = "".join([str(random.randint(0, 1)) for i in range(DATA_BITS_PER_BLOCK)])
    with open("key.txt", "w") as f:
        f.write(random_string)

def channel_estimation(symbols, known_block):

    # Take average value of H determined for each block
    symbols = np.average(symbols, axis=0)

    symbols_freq = np.fft.fft(symbols, N)
    print("imag", symbols_freq.imag)

    known_block_freq = np.fft.fft(known_block, N)
    print("imag", known_block_freq.imag)

    channel_response_freq = np.true_divide(
        symbols_freq,
        known_block_freq,
        out=np.zeros_like(symbols_freq),
        where=known_block_freq != 0,
    )

    # Remove DC value
    channel_response_freq[0] = 0
    channel_response_freq[int(N / 2)] = 0

    
    #Might be needed later to avoid decoding issues
    channel_response = np.fft.ifft(channel_response_freq, N)[:CP]
    plt.plot(channel_response)
    plt.show()
   
    # channel_response_freq = np.fft.fft(channel_response,N)
    
    return channel_response_freq

def receiver(data):
    # Normalise known symbol to 16 bit range
    known_symbol = np.array(get_known_data())
    known_symbol *= 1/np.max(known_symbol)

    # Normalise data to +- 1
    data = norm(data)

    chirp = sweep()
    
    shifts = shift_finder(chirp, data, SAMPLE_FREQUENCY,window=0)
    shift = shifts[0] + 1

    # Remove Stuff before and after data and split into frames
    data = data[shift - CHIRP_LENGTH * PREFIXED_SYMBOL_LENGTH:]
    data = [data[i : i + FRAME_LENGTH] for i in range(0, len(data), FRAME_LENGTH)]
    if len(data[-1]) != FRAME_LENGTH:
        del data[-1]

    # Check the power of each frame
    # Will be used to discard non-OFDM frames
    frame_powers = [np.sqrt(np.mean(np.square(frame))) for frame in data]
    #plt.plot(frame_powers)
    #plt.show()

    # Remove the chirp
    data = [frame[CHIRP_LENGTH * PREFIXED_SYMBOL_LENGTH :] for frame in data]

    # Channel Estimation
    frame = data[0]
    # Split into symbols
    frame = [frame[i : i + PREFIXED_SYMBOL_LENGTH] for i in range(0, len(frame), PREFIXED_SYMBOL_LENGTH)]
    # Isolate Estimation Symbols
    # Using the last 20 symbols might be more error-prone than useful --Charalambos
    # So use only first 20 symbols
    estimation_symbols = frame[:KNOWN_BLOCK_LENGTH]
    estimation_symbols = norm(estimation_symbols)
    
    # Check that symbols the same
    print(estimation_symbols[0][0:4])
    print(np.max(estimation_symbols))
    print(np.min(estimation_symbols))
    print(known_symbol[0:4])
    print(np.max(known_symbol))
    print(np.min(known_symbol))
    # assert (estimation_symbols[0] == known_symbol).all()
    # print(channel_response)
    channel_response = channel_estimation(estimation_symbols, known_symbol)
    plt.plot(channel_response.real)
    plt.plot(channel_response.imag)
    plt.show()
    
    # Isolate data symbols
    data = [symbol[CP:] for symbol in frame[KNOWN_BLOCK_LENGTH : - KNOWN_BLOCK_LENGTH] for frame in data]

    # FFT the symbols
    data = [np.fft.fft(symbol, N) for symbol in data]
    plt.scatter(np.array(data).real, np.array(data).imag)
    plt.show()

    # Divide each symbol by channel response
    data = [np.true_divide(symbol, channel_response).tolist() for symbol in data]

    # Discard second half of all symbols
    data = [symbol[1 : 1 + CONSTELLATION_VALUES_PER_BLOCK] for symbol in data]

    # Flatten into single list of symbols
    data = [symbol for frame in data for symbol in frame]
    plt.scatter(np.array(data[-1000:]).real, np.array(data[-1000:]).imag)
    plt.show()

    # Map each symbol to constellation values
    for i, symbol in enumerate(data):
        # Get distance to all symbols in constellation
        distances = {abs(symbol - value): key for key, value in CONSTELLATION.items()}
        # Get minimum distance
        minimum_distance = min(distances.keys())
        # Find symbol matching minimum distance and append
        data[i] = distances[minimum_distance]

    # Make into one big string
    data = "".join(["".join(symbol) for symbol in data])

    return data

def binary_to_text(input_data, print_out=False, save_to_file=True):
    # Convert to bytes
    output_data = [input_data[i : i + 8] for i in range(0, len(input_data), 8)]
    # Convert to bytearray
    output_data = bytearray([int(i, 2) for i in output_data])
    if save_to_file:
        with open("reciever_output.txt", "wb") as f:
            f.write(output_data)
    if print_out:
        print(output_data.decode())
    return output_data

# == CALLING THE FUNCTIONS == #
tx_data = transmit()

channel_response = [1,0.6,0.1,0.5, 0.9, 0.5, 0]

convolved_signal = sg.convolve(tx_data, channel_response)

convolved_signal = convolved_signal[:-(len(channel_response)-1)]

#rx_data = add_noise_amp(tx_data, 100)
rx_data = convolved_signal

data = receiver(rx_data)

data = xor_binary_and_key(data)
binary_to_text(data)
