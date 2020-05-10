import numpy as np
import matplotlib.pyplot as plt

# 1) 2 bits per constellation, gray coding pairs 00,01,11,10
# 2) IDFT N=1024, cyclic prefix K=32, information in freq bins 1 to 511
# 3) 2 bit groups added to the end of the data to complete an integer factor of 511 (All 00)
# 4) the long signal is made of blocks of 1056 values with 32 values at the beginning being the prefix  
# 5) Signal fed through ISI channel with an FIR impulse response in gr6channel
# 6) Added noise

## To demodulate:
# 1) Split into blocks of 1056
# 2) Discard cyclic prefixes
# 2.5) Multiply by e^jn(pi) to shift all 
#      lower freq back to their original place???
# 3) DFT 
# 4) Product with inverse FIR channel
# 5) Decide on optimal decoder for constellations


# print(len(data))      518496 data points -> 518496/1056 = 491


data = np.genfromtxt('gr6file.csv', delimiter='  ')
channel_response = np.genfromtxt('gr6channel.csv',delimiter='  ')

#Definitions

N = 1024        #data length
k = 32          #prefix length
L = len(data)   #signal length 
num_ofdm_symbols = int(L/(N+k))

fft_data = np.zeros([num_ofdm_symbols,N],complex)

demod_data = np.zeros([num_ofdm_symbols,N],complex)

const_data = np.zeros([num_ofdm_symbols,N],complex)


split_data = np.array_split(data,num_ofdm_symbols)  # 1) Split into blocks of 1056


# print(lenfft_data)

for i in range(num_ofdm_symbols):
    split_data[i] = split_data[i][k:]               # 2) Remove cyclic prefix


for i in range(len(split_data)):      
    fft_data[i] = np.fft.fft(split_data[i],N)       # 3) FFT of symbols
     
#printing channel impulse response
    
H_exact = np.fft.fft(channel_response, N)           #FFT of channel
# plt.plot(np.arange(N), abs(H_exact))
# plt.show()

for i in range(num_ofdm_symbols):
    demod_data[i] = np.true_divide(fft_data[i],H_exact)
    

    
mapping_table = {
    (0,0) :  1+1j,
    (0,1) : -1+1j,
    (1,0) : -1-1j,
    (1,1) :  1-1j,
}
for b1 in [0, 1]:
    for b0 in [0, 1]:
        B = (b1, b0)
        Q = mapping_table[B]
        #plt.plot(Q.real, Q.imag, 'bo')
        #plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
        #plt.show()
demapping_table = {v : k for k, v in mapping_table.items()}


    # array of possible constellation points
constellation = np.array([x for x in demapping_table.keys()])

## difference of abs distance of demod_data[0] with each item in constellation
for i in range(len(demod_data)):
 for j in range(N):    
  min_dist = 100
     
  for const in constellation:
    
      dist = np.abs(demod_data[i][j] - const)
    
      if dist < min_dist:
          min_dist = dist
          chosen_const = const
    
      const_data[i][j]= chosen_const

#Replace constellations with their mapping
raw_data = [demapping_table[const] for const in const_data[0]] 

#Convert from list of tuples to one large list
raw_data = [i for sub in raw_data for i in sub]

print(raw_data)