
import numpy as np
from scipy.io.wavfile import read

#Importing an example wave data set to experiment

data = read('clap.wav', mmap=False)
data = data[1]


## 1) Convert floats into binary (possibly wrong, only used for the example dataset)

def float_to_bin(inp):
    int32bits = np.asarray(inp, dtype=np.float32).view(np.int32).item() # item() optional
 return '{:032b}'.format(int32bits)

data = [float_to_bin(dat) for dat in data]
data = [char for num in data for char in num]

for i in range(len(data)):
    if data[i] == "-":
        data[i] = '0'

data = [int(num) for num in data]
data = np.array(data)

## 2) Map data to constellation symbols
def mapping(bits,const_length=2):
    """
    Takes:
        bits         : a numpy array of binary data
        const_length : constellation length (must be power of two)
    Returns:
        mapped_data  : a list of data mapped to constellations
    """
    const_map = {
        (0,0): complex(+1, +1),
        (0,1): complex(-1, +1),
        (1,1): complex(-1, -1),
        (1,0): complex(+1, -1),
    }

    #First Test that the constellation length for modulation is valid
    const_check = np.log2(const_length)
    if const_check != int(const_check):
        raise ValueError('Constellation length must be a power of 2.')

    #Divide into blocks equal to constellation length
    split_data = [tuple(bits[i * const_length:(i + 1) * const_length]) for i in range((len(bits) + const_length - 1) // const_length )]

    #Map the blocks of data into constellations
    mapped_data = [const_map[values] for values in split_data]

    return mapped_data

mapped_datas = mapping(data)  ##Testing array


## 3) Inverse FFT
def IFFT(mapped_data):
    return list(np.fft.ifft(mapped_data))

## 4) Split into blocks with given block_length, add given cyclic prefix
def organise(data, block_length = 1024, cp = 32):
    """
        Takes:
            data         : a list of mapped data
            block_length : desired length of OFDM symbols
            cp           : desired length of cyclic prefix
        Returns:
            block_data  : a list of data correctly formated in blocks with the cp appended
    """

    #Divides into blocks of length "block_length"
    block_data = [data[i * block_length:(i + 1) * block_length] for i in range((len(data) + block_length - 1) // block_length )]

    #Makes a list of all cyclic prefixes
    cyc = [const[-cp:] for const in block_data]

    #Loops and adds all cyclic prefixes into the beginning of each data block
    for i in range(len(block_data)):
        block_data[i].insert(0,cyc[i][0])

    return block_data

prefix_data = organise(IFFT(mapped_datas))  ###Testing array
###Unsure if we are supposed to do IFFT before splitting into blocks or after

### 5) Digital to Analog converter, acts as an interpolating filter
###    using a sinc function as p(t)

def DAC(pref_data,sample_rate):

    samples = len(pref_data)
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)

    carrier_sinc=np.sinc(time_array*np.pi/sample_rate)

    carrier_sinc = [complex(it) for it in carrier_sinc]

    pref_data = [complex(t) for tt in pref_data for t in tt]

    return np.convolve(pref_data,carrier_sinc) # Should this be 2D convolution?

dac_data = DAC(prefix_data,44100)


# 6) Modulate Data with carrier
# Upconvert to obtain the passband waveform
# Multiply be e^[j2pi(fc)k]  = cos(2pi(fc)k) + jsin(2pi(fc)k)

def modulate(dac_data,carrier_frequency,sample_rate):

    samples = len(dac_data)
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)
    carrier_signal_sin = np.sin(carrier_frequency * time_array * 2 * np.pi)
    carrier_signal_cos = np.cos(carrier_frequency * time_array * 2 * np.pi)

    mod_data = np.multiply(dac_data,(carrier_signal_cos+carrier_signal_sin))

    return np.real(mod_data)

g = modulate(dac_data,88200,44100)

with open("clap_data.csv", "w") as f:
    for i in range(len(g)):
        f.write(str(g[i])+ ",")
