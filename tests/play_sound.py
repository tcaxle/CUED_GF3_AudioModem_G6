"""
A simple script to play a sound of a certain frequency for a duration
"""

import numpy as np
import sounddevice as sd

FREQUENCY = 440
SAMPLE_RATE = 44100
DURATION = 1

# calulcate total number of samples
samples = int(SAMPLE_RATE * DURATION)

# produce numpy array of samples
time_array = np.linspace(0, DURATION, samples, False)

# Modulate the array with our frequency
note = np.sin(FREQUENCY * time_array * 2 * np.pi)

audio = np.hstack((note))

# normalize to 16-bit range
audio *= 32767 / np.max(np.abs(audio))

# convert to 16-bit data
audio = audio.astype(np.int16)

# start playback
sd.play(audio, SAMPLE_RATE)

# Wait for it to play
sd.wait()
