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

# Import data from given files
modulated_data = np.genfromtxt('gr6file.csv', delimiter='  ')
channel = np.genfromtxt('gr6channel.csv', delimiter='  ')

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
# 5.2) Minimum distance decode and map to bits
mapped_data = []
for block in unconvolved_data:
    minimum_distance_block = []
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
output_string = ""
for bit in mapped_data:
    output_string += str(bit)


def to_bytes(bits, size=8, pad='0'):
    chunks = [bits[n:n+size] for n in range(0, len(bits), size)]
    if pad:
        chunks[-1] = chunks[-1].ljust(size, pad)
    return bytearray([int(c, 2) for c in chunks])

x = to_bytes(output_string)
print(x[0:18])
del x[0:18]
print(x[0:18])
f = open('output.tiff', 'w+b')
f.write(x)
f.close()
                     
