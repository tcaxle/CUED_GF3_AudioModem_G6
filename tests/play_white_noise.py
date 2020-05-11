"""
A simple script to play a white noise
"""

import numpy as np
import sounddevice as sd

FREQUENCY = 440
SAMPLE_RATE = 44100
DURATION = 5

samples = SAMPLE_RATE * DURATION

time_array = np.linspace(0, DURATION, samples, False)

pure_signal = np.sin(FREQUENCY * time_array * 2 * np.pi)

noise_signal = np.random.normal(0, 1, pure_signal.shape)

total_signal = pure_signal + noise_signal

total_signal *= 32767 / np.max(np.abs(total_signal))

total_signal = total_signal.astype(np.int16)

sd.play(total_signal, SAMPLE_RATE)
sd.wait()
