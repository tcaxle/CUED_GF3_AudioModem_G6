"""
A simple script to record the sound from your microphone for three seconds and play it back at you.
"""
import sounddevice as sd
from scipy.io.wavfile import write, read

def record_sound(save=False,sample_rate=44100, duration=5):
    samples = int(sample_rate * duration)
    print("RECORDING")
    sound = sd.rec(samples, sample_rate, channels=1)
    sd.wait()
    print("END RECORDING")
    if save:
        write('rec_test.wav',sample_rate,sound)
    return sound

def play_sound(sound, sample_rate=44100):
    print("PLAYING")
    sd.play(sound, sample_rate)
    sd.wait()

data = read('clap.wav', mmap=False)
