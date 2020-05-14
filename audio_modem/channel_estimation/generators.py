from scipy.signal import chirp
import numpy as np
import sounddevice as sd

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
    print(f_sweep)
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
