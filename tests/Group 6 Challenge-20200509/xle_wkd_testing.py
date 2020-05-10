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
# 4) Convolve with inverse FIR channel
# 5) Decide on optimal decoder for constellations

import numpy as np

# Import data from given files
modulated_data = np.genfromtxt('gr6file.csv', delimiter='  ')
channel = np.genfromtxt('gr6channel.csv',delimiter='  ')

# 1) Split into blocks of 1056
modulated_data = np.array_split(modulated_data, 491)

# 2) Discard cyclic prefixes (first 32 bits)
modulated_data = [block[32:] for block in modulated_data]

# 3) DFT N = 1024
"""
NOT SURE IF THIS IS CORRECT
"""
demodulated_data = np.fft.fft(modulated_data, 1024)

# 4) Convolve with inverse FIR channel
# 4.1) Invert the channel
"""
DON'T KNOW HOW TO INVERT THE CHANNEL
"""
inverse_channel = channel
# 4.2) Convolve
"""
DON'T KNOW WHY CONVOLUTION ADDS 29 BITS TO DATA BLOCK (SEE PRINT)
"""
convolved_data = [np.convolve(block, inverse_channel) for block in demodulated_data]
print(len(convolved_data[0]))

# 5) Decide on optimal decoder for constellations
# 5.1) Define constellation
constellation = {
    complex(+1, +1) / np.sqrt(2): (0, 0),
    complex(-1, +1) / np.sqrt(2): (0, 1),
    complex(-1, -1) / np.sqrt(2): (1, 1),
    complex(+1, -1) / np.sqrt(2): (1, 0),
}
# 5.2) Minimum distance decode and map to bits
mapped_data = []
for block in convolved_data:
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

# 6) Decode binary data
# group data into bytes
mapped_data = [mapped_data[i:i+8] for i in range(0, len(mapped_data), 8)]
# convert lists of 8 1s and 0s to strings
bytes_array = []
for byte in mapped_data:
    byte_string = ""
    for bit in byte:
        byte_string += str(bit)
    bytes_array.append(int(byte_string, 2))
# convert strings to binary data
output_data = bytes(bytes_array)
# print binary data decoded UTF-8
print(output_data.decode())
