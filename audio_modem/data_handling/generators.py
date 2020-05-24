from scipy.signal import chirp
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Allow really big plots
mpl.rcParams['agg.path.chunksize'] = 10000

def sweep(duration=10, f_start=20, f_end=20000, sample_rate=44100, channels=1):
    """
    Plays a frequency sweep
    """
    # Calculate number of samples
    samples = int(duration * sample_rate)
    # Produce time array
    time_array = np.linspace(0, duration, samples)
    # Produce a frequency awway
    frequency_array = np.linspace(f_start, f_end, samples)
    # Produce frequency sweep
    f_sweep = chirp(time_array, f_start, duration, f_end)
    # Normalise sweep
    f_sweep *= 32767 / np.max(np.abs(f_sweep))
    f_sweep = f_sweep.astype(np.int16)
    # Play noise
    recording = sd.playrec(f_sweep, sample_rate, channels=channels)
    sd.wait()
    return f_sweep, recording, time_array, frequency_array

def white_noise(duration=5, centre_frequency=500, sample_rate=44100, channels=1):
    """
    Plays white noise
    """
    # Calculate number of samples
    samples = int(sample_rate * duration)
    # Produce time array
    time_array = np.linspace(0, duration, samples, False)
    # Produce signals
    pure_signal = np.sin(centre_frequency * time_array * 2 * np.pi)
    noise_signal = np.random.normal(0, 1, pure_signal.shape)
    noise_signal *= 32767 / np.max(np.abs(noise_signal))
    noise_signal = noise_signal.astype(np.int16)
    # Play signals
    recording = sd.playrec(noise_signal, sample_rate, channels=channels)
    sd.wait()
    return noise_signal, recording, time_array

def stepped_sweep(duration=10, steps=1000, f_start=20, f_end=20000, sample_rate=44100, channels=1):
    """
    Plays a stepped frequency sweep
    """
    # Calulcate number of samples
    samples = int(duration * sample_rate)
    # Calculate step width (in samples)
    step_samples = int(samples / steps)
    step_time = duration / steps
    # Produce sweep
    f_sweep = []
    time_array = []
    phase = 0
    for step_index in range(steps):
        # Get frequency
        frequency = f_start + step_index * (f_end - f_start) / steps
        omega = 2 * np.pi * frequency
        # Produce time array
        time_array = np.linspace(0, step_time, step_samples)
        # Produce signal
        f_sweep += np.sin(omega * time_array + phase).tolist()
    # Calulcate ending phase for smooth transition to next note
    phase = omega * (time_array[-1] + step_time / step_samples) + phase
    # Calculate time and frequency arrays
    time_array = np.linspace(0, duration, samples)
    frequency_array = np.linspace(f_start, f_end, steps)
    frequency_array = [frequency for sample in range(step_samples) for step in frequency_array]
    # Play and Record
    recording = sd.playrec(f_sweep, sample_rate, channels=channels)
    sd.wait()
    return f_sweep, recording, time_array, frequency_array
