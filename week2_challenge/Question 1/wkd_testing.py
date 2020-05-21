import numpy as np
import scipy.signal as sg
from scipy.io import wavfile

def wav_to_binary(input_file="input.wav"):
    """
    Parameters
    ----------
    input_file : FILE NAME (STRING)
        wave data stored as floats in a .wav file

    Returns
    -------
    output_data : STRING
        string binary data
    """

    # Ooen the file and read the data
    output_data = wavfile.read(input_file)[1]

    #Normalise Data
    norm_data = [i/32767 for i in output_data]

    # Add 1 to make all values positive
    # Then scale by 2^16 / 2 = 2^15
    # Then convert to integer (rounds down)
    # Now we have 32 bit integers

    norm_data = [int((datum + 1) * np.power(2, 15)) for datum in norm_data]

    # Now convert to binary strings
    # Use zfill to make sure each string is 16 bits long
    # (By default python would not include redundant zeroes)
    # (And that makes it super hard to decode)
    # And use "".join() to make the whole thing one big string
    return "".join(format(datum, "b").zfill(16) for datum in norm_data)
 
def sweep(duration=1, f_start=100, f_end=8000, sample_rate=48000, channels=1):
    """
    Plays a frequency sweep
    """
    # Calculate number of samples
    samples = int(duration * sample_rate)
    # Produce time array
    time_array = np.linspace(0, duration, samples)
    # Produce a frequency awway
    #frequency_array = np.linspace(f_start, f_end, samples)
    # Produce frequency sweep
    f_sweep = sg.chirp(time_array, f_start, duration, f_end)
    # Normalise sweep
    f_sweep *= 32767 / np.max(np.abs(f_sweep))
    
    f_sweep = f_sweep.astype(np.int16)
    # Play noise
    # print(f_sweep)
    # recording = sd.playrec(f_sweep, sample_rate, channels=channels)
    # sd.wait()
    return f_sweep, sample_rate
    


input_data = wav_to_binary('a7r56tu_received.wav')

chirp, data_rate = sweep()

known_ofdm_symbols = np.genfromtxt('a7r56tu_knownseq.csv', delimiter=',')

known_ofdm_symbols = [int(i) for i in known_ofdm_symbols]


