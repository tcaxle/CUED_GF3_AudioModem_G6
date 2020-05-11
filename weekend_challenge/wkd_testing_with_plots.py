# 1) 2 bits per constellation, gray coding pairs 00,01,11,10
# 2) IDFT N=1024, cyclic prefix K=32, information in freq bins 1 to 511
# 3) 2 bit groups added to the end of the data to complete an integer factor of 511 (All 00)
# 4) the long signal is made of blocks of 1056 values with 32 values at the beginning being the prefix
# 5) Signal fed through ISI channel with an FIR impulse response in gr6channel
# 6) Added noise

## To demodulate:
# 1) Split into blocks of 1056 (= 1024 + 32)
# 2) Discard cyclic prefixes
# 3) DFT
# 4) Unconvolve with inverse FIR channel
# 5) Minimum distance decode
# 6) Decode into text

import numpy as np
import matplotlib.pyplot as plt

# Import data from given files
modulated_data = np.genfromtxt('data.csv', delimiter='  ')
channel = np.genfromtxt('channel.csv', delimiter='  ')

# 0) Set up constants
N = 1024
CYCLIC_PREFIX = 32
block_length = N + CYCLIC_PREFIX
block_number = int(len(modulated_data) / block_length)

# 1) Split into blocks of 1056
modulated_data = np.array_split(modulated_data, block_number)

# 2) Discard cyclic prefixes (first 32 bits)
modulated_data = [block[CYCLIC_PREFIX:] for block in modulated_data]

# 3) DFT N = 1024
demodulated_data = np.fft.fft(modulated_data, N)

# 4) Convolve with inverse FIR channel
# 4.1) Invert the channel
inverse_channel = np.fft.fft(channel, N)
# 4.2) Convolve
unconvolved_data = [np.divide(block, inverse_channel) for block in demodulated_data]
# 4.3) Discard last half of each block
unconvolved_data = [block[1:512] for block in unconvolved_data]

# 5) Decide on optimal decoder for constellations
# 5.1) Define constellation
constellation = {
    complex(+1, +1): (0, 0),
    complex(-1, +1): (0, 1),
    complex(-1, -1): (1, 1),
    complex(+1, -1): (1, 0),
}

mapping = {
    (0,0): 1+1j,
    (0,1): -1+1j,
    (1,1): -1-1j,
    (1,0): 1-1j,
    }
for b1 in [0, 1]:
    for b0 in [0, 1]:
        B = (b1, b0)
        Q = mapping[B]
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
        plt.title('4 PSK contellation with Gray-Mapping')
        plt.plot(Q.real, Q.imag, 'bo', markersize=12)
        plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')



# 5.2) Minimum distance decode and map to bits
mapped_data = []
for block in unconvolved_data:
    minimum_distance_block = []
    plt.plot(block.real, block.imag, 'r+', markersize=1);
    for data_symbol in block:
        # Get distance to all symbols in constellation
        distances = {abs(data_symbol - constellation_symbol): constellation_symbol for constellation_symbol in constellation.keys()}
        # Get minimum distance
        minimum_distance = min(distances.keys())
        # Find symbol matching minimum distance
        symbol = distances[minimum_distance]
        # Append symbol data to mapped data array
        mapped_data.append(constellation[symbol][0])
        mapped_data.append(constellation[symbol][1])

# 6) Decode
# 6.1) Convert to byte strings
output_string = ""
for bit in mapped_data:
    output_string += str(bit)
output_data = [output_string[i : i + 8] for i in range(0, len(output_string), 8)]
# 6.2) Convert ints to bytearray
output_data = bytearray([int(i, 2) for i in output_data])
# 6.3 Remove first 18 items
#output_data = output_data[18:]
del output_data[0:18]
# 6.4) Write output file
with open("output.tiff", "w+b") as f:
    f.write(output_data)

