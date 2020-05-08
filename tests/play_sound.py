"""
A simple script to play a sound of a certain frequency for a duration
"""

import numpy as np
import sounddevice as sd

def play_sound(frequency=440, sample_rate=44100, duration=1):
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

    # start playback
    sd.play(note, sample_rate)

    # Wait for it to play
    sd.wait()

play_sound(440, 44100, 1)
play_sound(880, 44100, 1)
play_sound(220, 44100, 1)
