
import numpy as np

import itertools as it

import struct  #Used to convert sound wave floats to 32bit binary rep

from scipy.io.wavfile import read


data = read('clap.wav', mmap=False)

data = data[1]


#data = np.array([0,0,1,1,0,1,1,0,0,1,1,0]) #example data for testing
#channel = np.array([1, 0, 0.3+0.3j,0.2+0.2j]) 
#channel2 = np.genfromtxt('channel.csv', delimiter='  ') #example channel for testing
#data2=[]
def fun(inp): 
 int32bits = np.asarray(inp, dtype=np.float32).view(np.int32).item() # item() optional
 return '{:032b}'.format(int32bits)

data = [fun(dat) for dat in data]
data = [char for num in data for char in num]

for i in range(len(data)):
    if data[i] == "-":
        data[i] = '0'

data = [int(num) for num in data]        
data = np.array(data)
"""                         Float into bits is not working...
def floatToBits(f):
    s = struct.pack('>f', f)
    return np.binary_repr(struct.unpack('>l', s)[0])

y= [floatToBits(dat) for dat in data[1]]
"""

def mapping(bits,const_length=2):
  """
   Takes:
        bits         : a numpy array of binary data 
        const_length : constellation length (must be power of two)
   Returns:
        mapped_data  : a list of data mapped to constellations
   """
  const_map = {
    (0,0): 1+1j,
    (0,1): -1+1j,
    (1,1): -1-1j,
    (1,0): 1-1j,
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


#Inverse FFT 
def IFFT(mapped_data):
     return list(np.fft.ifft(mapped_data))


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


with open("clap_data.csv", "a+") as f:
    for i in range(len(prefix_data)):
        for j in range(len(prefix_data[i])):
            f.write(str(prefix_data[i][j]))


#Modulate Data with carrier
# def modulate(data,channel):
#     #Help, Stuck
#     np.convolve(data,channel)
    
# print(modulate(prefixed_data,channel))
