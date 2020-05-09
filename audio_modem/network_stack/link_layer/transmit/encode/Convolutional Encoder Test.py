# Convolutional Encoder attempt
#Example encoder with rate 1/3 having generator polynomials
#Proof of Concept
#G1 = (1,1,1)
#G2 = (0,1,1)
#G3 = (1,0,1)

import numpy as np

def conv_encode(input_data):
    #input_data must be an array of 1s and 0s
    
    s_0,s_1,s_2 = 0,0,0    #Initialise state inputs
    g_0,g_1,g_2 = 0,0,0    #Initialise state outputs
    output_data = np.array([]) #initialise output array

    for bit in range(len(input_data)):
        s_0 = input_data[bit]          #Updating the first bit

        g_0 = np.logical_xor(s_0,s_1)  #Convolving the first bit in two steps
        g_0 = np.logical_xor(g_0,s_2)

        g_1 = np.logical_xor(s_1,s_2)  #Convolving the second bit

        g_2 = np.logical_xor(s_0,s_2)  #Convolving the third bit

        output_data = np.append(output_data,[g_0,g_1,g_2]) #Append to output

        s_2 = s_1   #Shift values of s_2 and s_1 to their new values
        s_1 = s_0

    return output_data


test = np.array([0,0,1,0])
print(conv_encode(test))


#TODO: Update the function to take G1,G2,G3 and apply the logical XOR
#based on them instead of it being explicitely defined
#Appneding directly into a numpy array seems inefficient,
#find a better implementation?