import numpy as np
data = np.genfromtxt('gr6file.csv', delimiter='  ')
channel = np.genfromtxt('gr6channel.csv',delimiter='  ')



# 1) 2 bits per constellation, gray coding pairs 00,01,11,10
# 2) IDFT N=1024, cyclic prefix K=32, information in freq bins 1 to 511
# 3) 2 bit groups added to the end of the data to complete an integer factor of 511 (All 00)
# 4) the long signal is made of blocks of 1056 values with 32 values at the beginning being the prefix  
# 5) Signal fed through ISI channel with an FIR impulse response in gr6channel
# 6) Added noise

## To demodulate:
# 1) Split into blocks of 1056
# 2) Discard cyclic prefixes
# 3) DFT
# 4) Convolve with inverse FIR channel
# 5) Decide on optimal decoder for constellations


# print(len(data))      518496 data points -> 518496/1056 = 491

split_data = np.array_split(data,491)  # 1) Split into blocks of 1056

np.delete(split_data[0], 0:31)
