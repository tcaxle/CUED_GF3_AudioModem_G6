import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import matplotlib.pyplot as plt
import threading

sample_freq, sound = read('clap.wav',mmap=False)

def play_sound(sound= sound, sample_rate=44100):
    print("PLAYING")
    sd.play(sound, sample_rate)
    sd.wait()


def record_sound(save=True,sample_rate=44100, duration=5):
    samples = int(sample_rate * duration)
    print("RECORDING")
    sound = sd.rec(samples, sample_rate, channels=1)
    sd.wait()
    print("END RECORDING")
    if save:
        write('Aimilios_test.wav',sample_rate,sound)

    for point in sound:
        sound = sound.flatten()
        plt.specgram(sound, Fs=44100)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency(Hz)')
        plt.title('Spectrogram')
        plt.show()
    sd.wait()
    return sound

threading.Thread(target=play_sound).start()
threading.Thread(target=record_sound).start()
