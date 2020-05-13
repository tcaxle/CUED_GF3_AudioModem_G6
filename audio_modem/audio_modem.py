"""
The top level module for the project. Ties everything together
"""

# Imports:
## External
import numpy as np
from scipy.io import wavfile
from scipy.signal import bessel, lfilter, freqz, correlate
import sounddevice as sd
import matplotlib as mpl
from cmath import phase
mpl.rcParams['agg.path.chunksize'] = 10000
from matplotlib import pyplot as plt
## Internal

def transmit(output_audio=True, input_file="input.txt", output_to_file=False, output_file="data.wav", plot=False):
    """
    Takes input data and transmits it from your speakers.

    If output_to_file is set then it will produce a .wav file
    called "data.wav" (by default) instead of transmitting.

    If plot is set then it will produce a matplotlib plot of the output
    """

    """
    == INPUT ==
    """
    # Open input file
    with open(input_file, "rb") as f:
        data = f.read()

    """
    == ENCODING ==
    """
    # Encode text data as binary with utf-8 encoding
    data = "".join(format(datum, "b").zfill(8) for datum in data)
    # Group into symbol length chunks
    N = 2
    data = [data[i : i + N] for i in range(0, len(data), N)]
    print(len(data))

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
    # Split into blocks
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
    CYCLIC_PREFIX = 32
    data = [datum[-CYCLIC_PREFIX:] + datum for datum in data]

    # Combine back into one big list
    data = [symbol for block in data for symbol in block]

    """
    == MODULATION ==
    """
    # Produce sine-wave
    CARRIER_FREQUENCY = 10000
    carrier_omega = 2 * np.pi * CARRIER_FREQUENCY
    SYMBOL_FREQUENCY = 1000
    SAMPLE_FREQUENCY = 44100

    symbols = len(data)
    print(symbols)
    samples_per_symbol = int(SAMPLE_FREQUENCY / SYMBOL_FREQUENCY)
    samples = symbols * samples_per_symbol

    data_sin = []
    time_array = []
    for symbol_index in range(symbols):
        data_magnitude = abs(data[symbol_index])
        data_phase = phase(data[symbol_index])
        for sample_index in range(samples_per_symbol):
            time_index = symbol_index * samples_per_symbol + sample_index
            time = time_index / SAMPLE_FREQUENCY
            data_sin.append(data_magnitude * np.sin(carrier_omega * time + data_phase))
            time_array.append(time)

    data = data_sin

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


    """
    == OUTPUT TO FILE ==
    """
    if output_to_file:
        wavfile.write(output_file, SAMPLE_FREQUENCY, data)

    """
    == AUDIO TRANSMISSION ==
    """
    if output_audio:
        # Play data as audio
        # normalize to 16-bit range
        data *= 32767 / np.max(np.abs(data))
        # convert to 16-bit data
        data = data.astype(np.int16)
        # start playback
        sd.play(data, SAMPLE_FREQUENCY)
        sd.wait()

transmit(output_audio=False, output_to_file=True, plot=True)



def recieve(output_file="output.txt", input_from_file=False, input_file="data.wav"):
    """
    Records data from the microphone and recovers the information
    encoded within it.

    If input_from_file is set then it will take a .wav file
    called "data.wav" (by default) instead of using the microphone.
    """

    """
    For now, it holds the contents of wkd_tesing.py as it is transfered to various submodules
    """

    # Import data from input file
    sample_frequency, data = wavfile.read(input_file)

    """
    # Import channel from input file
    channel = np.genfromtxt('channel.txt', delimiter='  ')
    """


    """
    == QUADRATURE DEMODULATION ==
    """
    """
    == FILTER DESIGN ==
    """

    def bessel_lowpass_filter(data, cutoff_frequency, sample_frequency, order=4):
        nyquist_frequency = 0.5 * sample_frequency
        normal_cutoff = cutoff_frequency / nyquist_frequency
        b, a = bessel(order, normal_cutoff, btype="low", analog=False)
        y = lfilter(b, a, data)
        return y

    # Set up local oscillators
    CARRIER_FREQUENCY = 10000
    SYMBOL_FREQUENCY = 1000
    LO_FREQUENCY = 12000
    lo_omega = LO_FREQUENCY * 2 * np.pi
    difference_frequency = LO_FREQUENCY - CARRIER_FREQUENCY
    difference_omega = difference_frequency * 2 * np.pi

    samples = len(data)
    samples_per_symbol = int(sample_frequency / SYMBOL_FREQUENCY)
    LO_AMPLITUDE = 1

    lo = []
    time_array = []
    for sample_index in range(samples):
        time = sample_index / sample_frequency
        lo.append(LO_AMPLITUDE * np.sin(lo_omega * time))
        time_array.append(time)

    reference = []
    for sample_index in range(samples_per_symbol):
        time = sample_index / sample_frequency
        reference.append(np.sin(difference_omega * time))
    # Discard first 10 samples
    reference = reference[10:]

    #plt.plot(np.fft.fft(data), label="Input")
    #plt.plot(data, label="input")

    # Multiply by local oscillators
    data = [datum * lo_datum for datum, lo_datum in zip(data, lo)]

    #plt.plot(np.fft.fft(data_cos), label="After LO")

    # Low pass filter (bessel)
    CUTOFF_FREQUENCY = 10000
    data = bessel_lowpass_filter(data, CUTOFF_FREQUENCY, sample_frequency)

    # Split data into symbol chunks
    N = samples_per_symbol
    data = [data[i : i + N] for i in range(0, len(data), N)]
    # Discard first 10 samples in each symbol
    data = [datum[10:] for datum in data]

    def get_complex(datum, reference):
        # Magnitude
        magnitude = 1 / np.max(np.abs(datum))
        # Phase
        datum *= magnitude
        correlation = correlate(datum, reference)
        delta_time_array = np.linspace(-len(reference), len(reference), 2 * len(reference) - 1)
        time_shift = delta_time_array[correlation.argmax()] / sample_frequency
        phase_shift = 2 * np.pi * (((0.5 + time_shift * sample_frequency) % 1.0) - 0.5)
        return complex(np.cos(phase_shift), np.sin(phase_shift)) * magnitude
    data = [get_complex(datum, reference) for datum in data]

    #plt.plot(data[0], color="b")
    #plt.plot(reference, label="reference", color="r")
    #plt.title("Recieved Data")
    #plt.legend()
    #plt.show()

    """
    == DFT ==
    """
    # Split into blocks of 1056
    N = 1056
    data = [data[i : i + N] for i in range(0, len(data), N)]

    # 2) Discard cyclic prefixes (first 32 bits)
    CYCLIC_PREFIX = 32
    data = [block[CYCLIC_PREFIX:] for block in data]

    # 3) DFT N = 1024
    N = 1024
    data = [np.fft.fft(block, N) for block in data]

    """
    # 4) Convolve with inverse FIR channel
    # 4.1) Invert the channel
    inverse_channel = np.fft.fft(channel, N)
    # 4.2) Convolve
    unconvolved_data = [np.divide(block, inverse_channel) for block in demodulated_data]
    """
    # 4.3) Discard last half of each block
    data = [block[0:511] for block in data]

    # Flatten
    data = [symbol for block in data for symbol in block]

    # 5) Decide on optimal decoder for constellations
    # 5.1) Define constellation
    constellation = {
        "00" : complex(+1, +1) / np.sqrt(2),
        "01" : complex(-1, +1) / np.sqrt(2),
        "11" : complex(-1, -1) / np.sqrt(2),
        "10" : complex(+1, -1) / np.sqrt(2),
    }
    def minimum_distance_symbol(constellation, datum):
        distances = {abs(value - datum) : key for key, value in constellation.items()}
        minimum_distance = min(distances.keys())
        return distances[minimum_distance]

    data = [minimum_distance_symbol(constellation, datum) for datum in data]

    data = "".join(data)
    data = [data[i : i + 8] for i in range(0, len(data), 8)]
    data = bytearray([int(i, 2) for i in data])
    with open("output.txt", "wb") as f:
        f.write(data)

recieve(input_from_file=True)
