import sounddevice as sd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram

import generators

# Allow really big plots
mpl.rcParams['agg.path.chunksize'] = 10000

def plot_time_and_freq(waves, time_array=None):
    """
    Plot time and spectrum of ([wave], "label") objects
    """
    # Time domain
    plt.title("Time Domain")
    for wave, label in waves:
        plt.plot(time_array, wave, label=label)
    plt.legend()
    plt.show()


    # Frequency domain
    for wave, label in waves:
        wave = wave.flatten()
        plt.title(label)
        plt.specgram(wave, Fs=44100)
        plt.show()

def plot_sweep(duration=10, sample_rate=44100):
    input_wave, output_wave, time_array, frequency_array = generators.sweep(duration=10, f_start=20, f_end=20000, sample_rate=44100, channels=1)
    # Normalise
    input_wave = input_wave.astype(float)
    output_wave = output_wave.astype(float)
    input_wave *= 1 / np.max(np.abs(input_wave))
    output_wave *= 1 / np.max(np.abs(output_wave))

    # Time domain
    plot_time_and_freq(time_array=time_array, waves=[
        (input_wave, "input"),
        (output_wave, "output"),
    ])

    # Frequency Domain


plot_sweep()
