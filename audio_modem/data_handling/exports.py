import numpy as np
import soundevice as sd
from scipy.io import wavfile


def play_output(input_data):
    """
    Parameters
    ----------
    input_data : LIST of LIST of FLOAT
        list of transformed blocks with cyclic prefix
    
    * Normalises data to +/- 1.0
    * Transmits data from audio device
    """
   
    # convert to 16-bit data
    data = np.array(input_data).astype(np.float32)
    # Normalise to 16-bit range
    data *= 32767 / np.max(np.abs(data))
    # start playback

    sd.play(data)
    sd.wait()
     
    pass

def wav_output(input_data,sample_frequency):
    """
    Parameters
    ----------
    input_data : LIST of LIST of FLOAT
        list of transformed blocks with cyclic prefix
    sample_frequency : INT
        Sample frequency of the data to be saved
        
    * Normalises data to +/- 1.0 if needed
    """
    if np.max(np.abs(input_data)) > 1.0:
        
        print("The data was not normalised for int16")
        
        data = np.array(input_data).astype(np.float32)
        
        data *= 32767 / np.max(np.abs(data))      
    
    data = input_data.astype(np.int16)
    
    wavfile.write('sent.wav',sample_frequency,data)
        
    ### sd.rec method might be better
    pass

def text_output(input_data):
    """

    Parameters
    ----------
    input_data : LIST of LIST of FLOAT
        list of transformed blocks with cyclic prefix
    
    Outputs a text file of the provided data
    """
    # convert to 16-bit data
    data = np.array(input_data).astype(np.float32)
    # Normalise to 16-bit range
    data *= 32767 / np.max(np.abs(data))
    
    with open('output.txt', 'w') as f:
        for i in data:
            f.write(str(i) + ',')

    pass