
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import scipy.signal as sg
PLOTTING = False


## 1) Convert from floats to binary string
## NB: wav data bounded between -1 and 1
## NB: floats are only 16 bits (doubles are 32)

def float_to_bin(data):
    """
    Parameters
    ----------
    data : ndarray,list
        Imported wav file (int16), written data,etc
    
    Returns
    -------
    LIST
        Converts a string of floats into ints (round down)
        then into binary and returns it as a large list.
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
    return "".join(format(datum, "b").zfill(16) for datum in data)

## 2) Map data to constellation symbols

def mapping(bit_string,const_length=2):
    '''
    Parameters
    ----------
    bit_string : LIST
        A list containing binary values
    const_length : INT, optional
        Length of constellations. Must be a power of 2.
        Other lengths besides 2 not implemented here.
        The default is 2.
        
    Returns
    -------
    mapped_data : LIST
        List containing mapped constellation symbols in QPSK using
        Gray Mapping

    '''
    
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


## 4) Split into blocks with given block_length, IFFT and then add given cyclic prefix

def organise(data, block_length = 1024, cp = 512):
    """
    Parameters
    ----------
    data : LIST
        Takes a list of mapped constellation symbols
    block_length : INT, optional
        Length of the blocks in which the symbols will be divided.
        The default is 1024.
    cp : INT, optional
        Length of the cyclic prefix to be prepended into each
        block. The default is 512.

    Returns
    -------
    block_data : LIST of LISTS
        First, the data is split into blocks of a defined length.
        Then, in each block, the cyclic prefix is prepended.
        block_data is the returned list containing the output of 
        the above operations.
    """
    
    ###This line is in contrast to the white papers
    #The white papers say to use a 1024 DFT here, setting bins 0 and 512 to '0'
    #and dividing the data between bins 1-511. Then find the reverse complex
    #conjugate of data in 1-511 and add to bin 513-1023. We are not doing this
    #at the moment.
    
    #Divides into blocks of length "block_length"
    block_data = [data[i : i + block_length] for i in range(0, len(data), block_length)]
    
    """
    #THIS SUPPOSEDLY MAKES THE DATA REAL AFTER IFFT    
    #Multiplies the latter half of the block length by the negative complex conjugate to make all the data real
    
    #Implementation A
    
    for i in range(len(block_data)):
        for j in range(len(block_data[j])):
            if j > (block_length/2):
                block_data[i][j] = np.conj(block_data[i][block_length-j])
        block_data[i][0] = 0
        block_data[i][1] = 0
        block_data[i][int(block_length/2)] = 0
        block_data[i][block_length-1] = 0

    """
    
    #Implementation B

    def latter_half_conjugate(block):
        if len(block) % 2 != 0:
            raise Exception("OFDM block must have even length, preferably a power of 2")
        mid = int(len(block) / 2)
        block[mid:] = [np.conj(datum) for datum in block[mid:]]
        block[0], block[1], block[mid], block[-1] = 0, 0, 0, 0
        return block
    block_data = [latter_half_conjugate(block) for block in block_data]
    
    
    #Does the IFFT on each block in the data
    ifft_data = [list(np.fft.ifft(block,block_length)) for block in block_data]

    #Adds cyclic prefixes
    block_data = [np.hstack([block[-cp:],block]) for block in ifft_data] 

    ### The above line does what the following commented code does, but in one line
    ### The code below is kept for ease of reading/understanding
    
    # for block in ifft_data:
    #     cyc = block[-cp:]
    #     block = np.hstack([cyc,block])
        
    return block_data
   

### 5) Digital to Analog converter, acts as an interpolating filter
###    using a sinc function as p(t)

def DAC(pref_data,sample_rate):
    """
    Parameters
    ----------
    pref_data : LIST
        Data that's already been through IFFT 
        and has CP prepended in each block
    sample_rate : INT
        Sample rate of the data

    Returns
    -------
    LIST
        Returns a list that has the data convolved with a sinc function
    """

    samples = len(pref_data[0])
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)

    carrier_sinc=np.sinc(time_array*np.pi/sample_rate)

    carrier_sinc = [complex(it) for it in carrier_sinc]

    return [np.convolve(block,carrier_sinc,'same') for block in pref_data]   



# 6) Modulate Data with carrier
# Upconvert to obtain the passband waveform
# Multiply be e^[j2pi(fc)k]  = cos(2pi(fc)k) + jsin(2pi(fc)k)

def modulate(dac_data,carrier_frequency,sample_rate):
    """
    Parameters
    ----------
    dac_data : LIST of LISTS
        A list containing OFDM blocks with added CPs
    carrier_frequency : INT
        Carrier frequency of the modulation to be done
    sample_rate : INT
        Sampling rate of the data, same as sampling frequency

    Returns
    -------
    LIST of LISTS
        Data in blocks after being modulated by a sinc pulse

    """

    samples = len(dac_data[0])
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)
    carrier_signal_sin = np.sin(carrier_frequency * time_array * 2 * np.pi)
    carrier_signal_cos = np.cos(carrier_frequency * time_array * 2 * np.pi)

    mod_data = [np.multiply(blocks,(carrier_signal_cos+carrier_signal_sin)) for blocks in dac_data]

    return list(np.real(mod_data))




if __name__ == "__main__":
    
# Importing an example wave data set to experiment
    SYNCHRO = True
    
    sample_frequency, data = read('clap.wav', mmap=False)
    
    bin_data = float_to_bin(data)

    mapped_datas = mapping(bin_data) 
     
    prefix_data = organise(mapped_datas)
  
    if SYNCHRO:
        
        prefix_data = np.array(prefix_data)
        
        # def sweep(duration=512/44100, f_start=20, f_end=1500, sample_rate=44100, channels=1):
        #     """
        #     Plays a frequency sweep
        #     """
        #     # Calculate number of samples
        #     samples = int(duration * sample_rate)
        #     # Produce time array
        #     time_array = np.linspace(0, duration, samples)
        #     # Produce a frequency awway
        #     frequency_array = np.linspace(f_start, f_end, samples)
        #     # Produce frequency sweep
        #     f_sweep = sg.chirp(time_array, f_start, duration, f_end)
        #     # Normalise sweep
        #     #f_sweep *= 32767 / np.max(np.abs(f_sweep))
            
        #     #f_sweep = f_sweep.astype(np.int16)
   
        #     return f_sweep/3

        #print(len(prefix_data[0])) #Numpy Array with Numpy Arrays
        prefix_data = list(prefix_data)
        
        preamble = np.sqrt(2)*prefix_data[5][:1024]
       
        preamble[::2] = 0 
        
        preamble = np.hstack((preamble[-512:],preamble))
        
        prefix_data.insert(0,preamble)
        
        prefix_data = np.array(prefix_data)
        
        comp_noise = (np.random.randn(*prefix_data.shape)+1j*np.random.randn(*prefix_data.shape))        
        
        prefix_data = 0.001*comp_noise + prefix_data
                
        prefix_data = list(prefix_data)
        
        x = np.array([0 for i in range(1536)])
    
        #print(len(prefix_data[0]))
        prefix_data.insert(0,x)
        #print(len(prefix_data[0]))
        
        #np.savetxt("ofdm.csv",prefix_data[0:8],delimiter=',')
        with open('ofdm_data4.csv', 'w') as f:
            for i in range(8):
                for j in prefix_data[i]:
                    f.write(str(j) + ',')
    #%%    
    dac_data = DAC(prefix_data,44100)

    g = modulate(dac_data,88200,44100)
    
    g = [point for block in g for point in block]
    
    #np.savetxt("clap_data.csv", g, delimiter=",")
        
    if PLOTTING:
        
    # Plot prefix_data in the time domain
    # x axis = 0-7.5 seconds
    # y axis = absolute value of prefix data
        
        sample_length = len(data)
        duration = sample_length/16/sample_frequency
        cp_length = len(prefix_data)*len(prefix_data[0])
        cp_duration = cp_length/sample_length*duration
        
        x = np.linspace(0,cp_duration,cp_length)
        y = [dat for prefix in prefix_data for dat in np.abs(prefix)]
        plt.plot(x,y)
        plt.title("Data after IFFT and with prepended CP")
        plt.ylabel('Magnitude (N/A)')
        plt.xlabel('time (s)')
        plt.show()


        
    #Plot 100 symbols
        x = np.arange(1,100)
        plt.plot(x,y[151:250])
        

    #Same plot with noise
        print(np.average(y))
        noise = np.random.normal(0,0.005,len(y))
        ns_data = [noise[a]+y[a] for a in range(len(y))]
        plt.plot(x,ns_data[151:250],'r')
        plt.show()