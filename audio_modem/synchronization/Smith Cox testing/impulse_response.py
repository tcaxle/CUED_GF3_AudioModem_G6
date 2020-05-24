import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import threading
import matplotlib.pyplot as plt




#SENDS AN IMPULSE AFTER A DELAY OF 2 SECONDS


def send_impulse(frequency= 440, sample_rate=44100, duration=0.03):
    # calulcate total number of samples
    samples = int(sample_rate * duration)

    # produce numpy array of samples
    time_array = np.linspace(0, duration, samples, False)

    # Modulate the array with our frequency
    note = np.sin(frequency * time_array * 2 * np.pi)
    
    note = np.concatenate((np.zeros(2*44100), note))


    # normalize to 16-bit range
    note *= 62767 / np.max(np.abs(note))

    # convert to 16-bit data
    note2 = note.astype(np.int16)

    # start playback
    sd.play(note2, sample_rate)

    # Wait for it to play
    sd.wait()
    time_array = np.linspace(0, 2.03, samples+88200, False)
    
    # plt.plot(time_array, note)
    # plt.show
    plt.subplot(211)
    plt.plot(time_array, note)
    #plt.show()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (Arbitrary')
    plt.ylim(0,70000)
    plt.xlim(0,10)

    return time_array, note
    

def record_sound(save=False,sample_rate=44100, duration=10):
    samples = int(sample_rate * duration)
    print("RECORDING")
    sound = sd.rec(samples, sample_rate, channels=1)
    sd.wait()
    print("END RECORDING")
    if save:
        write('compimp.wav',sample_rate,sound)
        
  
    
    x = np.linspace(0, 10, samples)
    plt.subplot(212)
    plt.ylim(0,60000)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (Arbitrary')

    plt.plot(x, 12767*sound)
    plt.show()
    return samples, sound

# send_impulse(500, 44100, 0.2)

threading.Thread(target= record_sound).start()
threading.Thread(target= send_impulse).start()








