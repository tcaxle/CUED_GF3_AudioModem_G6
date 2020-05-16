"""
A simple script to play a sound of a certain frequency for a duration
"""

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

def sent_sound(frequency=440, sample_rate=44100, duration=5):
    # calulcate total number of samples
    samples = int(sample_rate * duration)

    # produce numpy array of samples
    time_array = np.linspace(0, duration, samples, False)

    # Modulate the array with our frequency
    note = np.sin(frequency * time_array * 2 * np.pi)


    # normalize to 16-bit range
    note *= 32767 / np.max(np.abs(note))

    # convert to 16-bit data
    note = note.astype(np.int16)
    
    write("check.wav",sample_rate,note)
    # Wait for it to play
    sd.wait()

sent_sound(880, 44100, 5)


def rec_sound(frequency=440, sample_rate=44100, duration=5):
    
    samples = int(sample_rate * duration)
    
    time_array = np.linspace(0, duration, samples, False)
    
    note = np.sin(frequency * time_array * 2 * np.pi)

    note *= 32767 / np.max(np.abs(note))

    note = note.astype(np.int16)
    
    
    ##Generate white noise

    DURATION = 3
    
    wn_time_array = np.linspace(0, DURATION, samples, False)
    
    car_note = np.sin(frequency * wn_time_array * 2 * np.pi)

    noise_signal = np.random.normal(0, 1, car_note.shape)

    noise_signal *= 32767 / np.max(np.abs(noise_signal))

    noise_signal = noise_signal.astype(np.int16)
    
    ##Add white noise to the begginning
    
    beg_note = np.concatenate((noise_signal,note))
    
    write("beg_rec.wav",sample_rate,beg_note)  
    
    ##Add white noise to the end
    
    end_note = np.concatenate((note,noise_signal))
    
    write("end_rec.wav",sample_rate,end_note)
    
    #Add noise to both start and end
    
    both_note = np.concatenate((beg_note,noise_signal))
    
    write("both_rec.wav",sample_rate,both_note)

rec_sound(880)
    


    
