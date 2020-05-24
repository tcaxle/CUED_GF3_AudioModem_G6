import meatplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.signal as sg
from cmath import phase
#from scipy.io.wavfile import write

def tx_chirp(f0=500,f1=2000,sample_rate=44100, duration=3,plot=True,output_audio = True):

    # calulcate total number of samples
    samples = int(sample_rate * duration)

    # produce numpy array of samples
    time_array = np.linspace(0, duration, samples, False)

    # Create chirp
    data = sg.chirp(time_array,f0,duration,f1,method='linear')
    
        # Play data as audio
        # normalize to 16-bit range
        # data *= 32767 / np.max(np.abs(data))
        # convert to 16-bit data
        # data = data.astype(np.int16)
        # start playback
    sd.play(data, sample_rate)
    sd.wait()

tx_chirp()