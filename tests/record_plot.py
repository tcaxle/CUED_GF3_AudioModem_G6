"""
A simple script to record the sound from your microphone for three seconds and play it back at you.
"""
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import scipy.fftpack
import numpy as np

def record_sound(save=False,sample_rate=44100, duration=5):
    samples = int(sample_rate * duration)
    print("RECORDING")
    sound = sd.rec(samples, sample_rate, channels=1)
    sd.wait()
    print("END RECORDING")
    print(samples)
    print(sample_rate)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    x = np.linspace(0, 5, samples)
    plt.plot(x, sound)
    plt.show()
    return samples, sound

# record_sound(True,44100,5)

num_samples, sound = record_sound(True)


CENTRE_FREQUENCY = 500
SAMPLE_RATE = 44100
DURATION = 5

samples = SAMPLE_RATE * DURATION

time_array = np.linspace(0, DURATION, samples, False)

pure_signal = np.sin(CENTRE_FREQUENCY * time_array * 2 * np.pi)

noise_signal = np.random.normal(0, 0.1, pure_signal.shape)

#noise_signal *= 32767 / np.max(np.abs(noise_signal))

#noise_signal = noise_signal.astype(np.int16)



    
def plot_FFT(num_samples, sound, sample_rate=44100):
    
    # N = 2000
    # t = 1/44100
    # x = np.linspace(0.0,N*t,N)
    # y = np.sin(600*10**3*np.pi*x)
    # f = np.fft.fft(y)
    
    N = num_samples
    t = 1/sample_rate
    y = sound
    f = np.fft.fft(y)
    
    xf = np.linspace(0.0, 1.0/(2.0*t), N//2)
    
    plt.ylim(-5.5,5.5)
    plt.plot(xf ,2.0/N*np.abs(f[0:N//2])) ## will show a peak at a frequency of 1 as it should.
    plt.show()
    

  
def play_sound(sound, sample_rate=44100):
    print("PLAYING")
    sd.play(sound, sample_rate)
    sd.wait()


plot_FFT(samples, noise_signal)

