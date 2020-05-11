
import numpy as np

import itertools as it



# Convert bytes to bits and split bits to blocks (IF NEEDED, Unused for now)
def integer_to_bitarray(ints, size):
    
    vectorized_binary_repr = np.vectorize(np.binary_repr)
            
    binary_words = vectorized_binary_repr(np.array(ints, ndmin=1), size)
    
    return np.fromiter(it.chain.from_iterable(binary_words), dtype=np.int8)

def bytes_to_bits(data):
    #To Do
    return True


data = np.array([0,0,1,1,0,1,1,0,0,1,1,0]) #example data for testing

#3) Constellation Map



    
def modulation(bits,const_length=2):
    
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
    
    #split_data = [tuple(i) for i in split_data]
    
    #Map the blocks of data into constellations
    mapped_data = [const_map[values] for values in split_data]
       
    return mapped_data
