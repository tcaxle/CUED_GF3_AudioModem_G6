import sounddevice as sd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import chirp

import generators

# Allow really big plots
mpl.rcParams['agg.path.chunksize'] = 10000

def plot_sweep(duration=10, sample_rate=44100):
    input_wave, output_wave, time_array, frequency_array = generators.sweep(duration=10, f_start=20, f_end=20000, sample_rate=44100, channels=1)
    # Normalise
    input_wave = input_wave.astype(float)
    output_wave = output_wave.astype(float)
    input_wave *= 1 / np.max(np.abs(input_wave))
    output_wave *= 1 / np.max(np.abs(output_wave))
    plt.plot(frequency_array, input_wave, label="input")
    plt.plot(frequency_array, output_wave, label="ouput")
    plt.xscale("log")
    plt.legend()
    plt.show()

plot_sweep()
