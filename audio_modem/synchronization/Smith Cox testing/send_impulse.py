"""
A simple script to play a sound of a certain frequency for a duration
"""

import numpy as np
import sounddevice as sd

#SENDS AN IMPULSE AFTER A DELAY OF 2 SECONDS


def send_impulse(frequency= 440, sample_rate=44100, duration=1):
    # calulcate total number of samples
    samples = int(sample_rate * duration)

    # produce numpy array of samples
    time_array = np.linspace(0, duration, samples, False)

    # Modulate the array with our frequency
    note = np.sin(frequency * time_array * 2 * np.pi)
    
    note = np.concatenate((np.zeros(2*44100), note))


    # normalize to 16-bit range
    note *= 32767 / np.max(np.abs(note))

    # convert to 16-bit data
    note = note.astype(np.int16)

    # start playback
    sd.play(note, sample_rate)

    # Wait for it to play
    sd.wait()


send_impulse(500, 44100, 0.2)
