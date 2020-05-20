import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.signal as sg
from cmath import phase
#from scipy.io.wavfile import write

def tx_chirp(f0=500,f1=2000,sample_rate=44100, duration=3,plot=True,output_audio = True):

    # calulcate total number of samples
    samples = int(sample_rate * duration)

    # produce numpy array of samples
    time_array = np.linspace(0, duration, samples, False)

    # Create chirp
    data = sg.chirp(time_array,f0,duration,f1,method='linear')

    data = [int((datum + 1) * np.power(2, 15)) for datum in data]
    # Now convert to binary strings
    # Use zfill to make sure each string is 16 bits long
    # (By default python would not include redundant zeroes)
    # (And that makes it super hard to decode)
    # And use "".join() to make the whole thing one big string
    data = "".join(format(datum, "b").zfill(16) for datum in data)

    #split into blocks of 2
    data = [data[i : i + 2] for i in range(0, len(data), 2)]
    #Final element is 0 instead of 00, removed
    data = data[:-2]
    """
    == CONSTELLATION MAPPING ==
    """
    # Map onto symbols
    constellation = {
        "00" : complex(+1, +1) / np.sqrt(2),
        "01" : complex(-1, +1) / np.sqrt(2),
        "11" : complex(-1, -1) / np.sqrt(2),
        "10" : complex(+1, -1) / np.sqrt(2),
    }

    data = [constellation[datum] for datum in data]

    """
    == INVERSE DFT ==
    """
    # Split into constellation blocks
    N = 511
    data = [data[i : i + N] for i in range(0, len(data), N)]

    """
    # Make last block up to length N with 0s
    for i in range(N - len(data[-1])):
        data[-1].append(constellation["00"])
    """

    # Inverse DFT
    N = 1024
    data = [list(np.fft.ifft(datum, N)) for datum in data]

    # Cyclic Prefix
    CYCLIC_PREFIX = 1024
    data = [datum[-CYCLIC_PREFIX:] + datum for datum in data]

    # Combine back into one big list
    data = [symbol for block in data for symbol in block]

    """
    DAC
    """

    carrier_sinc=np.sinc(time_array*np.pi/sample_rate)

    carrier_sinc = [complex(it) for it in carrier_sinc]

    data = [complex(t) for t in data]

    data = sg.fftconvolve(data,carrier_sinc) # Convolved data with sinc as p(t)


    """
    == MODULATION ==
    """

    carrier_frequency = 10000
    samples = len(data)
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)
    carrier_signal_sin = np.sin(carrier_frequency * time_array * 2 * np.pi)
    carrier_signal_cos = np.cos(carrier_frequency * time_array * 2 * np.pi)

    mod_data = np.multiply(data,(carrier_signal_cos+carrier_signal_sin))

    data = np.real(mod_data)



    # Convert to numpy array
    data = np.array(data)

    """
    == PLOT ==
    """
    if plot:
        plt.plot(time_array, data)
        plt.title("Transmitted Data")
        plt.legend()
        plt.show()


    # """
    # == OUTPUT TO FILE ==
    # """
    # if output_to_file:
    #     wavfile.write(output_file, SAMPLE_FREQUENCY, data)

    """
    == AUDIO TRANSMISSION ==
    """
    # if output_audio:
    #     # Play data as audio
    #     # normalize to 16-bit range
    #    # data *= 32767 / np.max(np.abs(data))
    #     # convert to 16-bit data
    #    # data = data.astype(np.int16)
    #     # start playback
    #     sd.play(data, sample_rate)
    #     sd.wait()

tx_chirp()
