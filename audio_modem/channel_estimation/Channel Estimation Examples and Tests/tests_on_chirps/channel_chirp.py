import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import matplotlib.pyplot as plt
import threading
import numpy as np
import scipy.signal as sg
from scipy.signal import chirp, spectrogram


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
    


def record_sound(save=True, sample_rate=44100, duration=10):
    samples = int(sample_rate * duration)
    print("RECORDING")
    sound = sd.rec(samples, sample_rate, channels=1)
    sd.wait()
    print("END RECORDING")
    if save:
        write('test.wav',sample_rate,sound)

    #for point in sound:
     #   sound = sound.flatten()
      #  plt.specgram(sound, Fs=44100)
      #  plt.xlabel('Time (s)')
      #  plt.ylabel('Frequency(Hz)')
       # plt.title('Spectrogram')
       # plt.show()
    sd.wait()
    return sound

threading.Thread(target=record_sound).start()
threading.Thread(target=sweep).start()
