"""
A simple script to play a white noise
"""

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
DURATION = 5

samples = SAMPLE_RATE * DURATION

time_array = np.linspace(0, DURATION, samples, False)

noise_signal = np.random.normal(0, 1, pure_signal.shape)

noise_signal *= 32767 / np.max(np.abs(noise_signal))

noise_signal = noise_signal.astype(np.int16)

sd.play(noise_signal, SAMPLE_RATE)
sd.wait()
