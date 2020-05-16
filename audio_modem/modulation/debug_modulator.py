
import numpy as np
from scipy.io.wavfile import read

# Importing an example wave data set to experiment

sample_frequency, data = read('clap.wav', mmap=False)
"""
Convert from floats to binary string
NB: wav data bounded between -1 and 1
NB: floats are only 16 bits (doubles are 32)
"""
# Add 1 to make all values positive
# Then scale by 2^16 / 2 = 2^15
# Then convert to integer (rounds down)
# Now we have 32 bit integers
data = [int((datum + 1) * np.power(2, 15)) for datum in data]
# Now convert to binary strings
# Use zfill to make sure each string is 16 bits long
# (By default python would not include redundant zeroes)
# (And that makes it super hard to decode)
# And use "".join() to make the whole thing one big string
data = "".join(format(datum, "b").zfill(16) for datum in data)

print("")
print("First 20 input bits:")
print(data[:20])
print('Lenght: ' ,len(data))


## 2) Map data to constellation symbols
def mapping(bit_string,const_length=2):
    """
    Takes:
        bits         : a numpy array of binary data
        const_length : constellation length (must be power of two)
    Returns:
        mapped_data  : a list of data mapped to constellations
    """
    const_map = {
        "00": complex(+1, +1)/np.sqrt(2),
        "01": complex(-1, +1)/np.sqrt(2),
        "11": complex(-1, -1)/np.sqrt(2),
        "10": complex(+1, -1)/np.sqrt(2),
    }

    #First Test that the constellation length for modulation is valid
    const_check = np.log2(const_length)
    if const_check != int(const_check):
        raise ValueError('Constellation length must be a power of 2.')

    #Divide into blocks equal to constellation length
    # Use array slicing on strings for array of strings
    split_data = [bit_string[i : i + const_length] for i in range(0, len(bit_string), const_length)]

    #Map the blocks of data into constellations
    mapped_data = [const_map[values] for values in split_data]

    return mapped_data

mapped_datas = mapping(data)  ##Testing array

print("")
print("Mapped in 10 constellation symbols:")
print(mapped_datas[:10])
print('Length: ' ,len(mapped_datas))


BLOCK_LENGTH = 1024

# 3) Divides into blocks of length "block_length"
block_data = [mapped_datas[i * BLOCK_LENGTH:(i + 1) * BLOCK_LENGTH] for i in range((len(mapped_datas) + BLOCK_LENGTH - 1) // BLOCK_LENGTH )]


print("")
print("After Splitting:")
print(block_data[0][:10])
print('Length:', len(block_data))

## 4) Inverse FFT
def IFFT(mapped_data,N):
    return list(np.fft.ifft(mapped_data,N))

ifft_data = [IFFT(block,1024) for block in block_data]


print("")
print("After IFFT:")
print(ifft_data[:10])
print('Length: ' ,len(ifft_data))


### Now we have blocks of length N that have been passed through IDFT of length N

### 5) ADD Cyclic Prefix


CP = 512

def addCP(ofdm_block,cyc_p):
    cp = ofdm_block[-cyc_p:]               # take the last CP samples ...
    return np.hstack([cp, ofdm_block])  # ... and add them to the beginning

prefixed_data = [addCP(block,CP) for block in ifft_data]


print("")
print("After prefix:")
print(prefixed_data[:10])
print('Length: ' ,len(prefixed_data))
print('CP + Block Length: ', len(prefixed_data[0]))



### 6) Digital to Analog converter, acts as an interpolating filter
###    using a sinc function as p(t)
### Join all blocks in a large list and then convolve with sinc for DAC

def DAC(pref_data,sample_rate):

    samples = len(pref_data[0])
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)

    carrier_sinc=np.sinc(time_array*np.pi/sample_rate)

    carrier_sinc = [complex(it) for it in carrier_sinc]

    conv_list = [np.convolve(block,carrier_sinc,'same') for block in pref_data]   

    return conv_list

dac_data = DAC(prefixed_data,44100)


print("")
print("DAC using a sinc pulse:")
print(dac_data[:10])
print('Length: ' ,len(dac_data))
print('DAC Block Length: ', len(dac_data[0]))



# 7) Modulate Data with carrier
# Upconvert to obtain the passband waveform
# Multiply be e^[j2pi(fc)k]  = cos(2pi(fc)k) + jsin(2pi(fc)k)

def modulate(dac_data,carrier_frequency,sample_rate):

    samples = len(dac_data[0])
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)
    carrier_signal_sin = np.sin(carrier_frequency * time_array * 2 * np.pi)
    carrier_signal_cos = np.cos(carrier_frequency * time_array * 2 * np.pi)

    mod_data = [np.multiply(blocks,(carrier_signal_cos+carrier_signal_sin)) for blocks in dac_data]

    return np.real(mod_data)

g = modulate(dac_data,88200,44100)

print("")
print("Modulate data:")
print(g[:10])
print('Length: ' , len(g))
print('Mod Block Length: ', len(g[0]))

### 8) create a continuous list of data
output = [point for block in g for point in block]

print(output[:10])
print(len(output))
'''
with open("clap_data.csv", "w") as f:
    for i in range(len(g)):
        f.write(str(g[i])+ ",")
        
'''
