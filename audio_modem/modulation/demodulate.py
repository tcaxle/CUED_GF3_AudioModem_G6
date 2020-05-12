"""
A module for demodulating input data recieved by the channel
"""

import numpy as np

def demodulate(modulated_data, carrier_frequency, sample_rate):
    """
    Takes:
        input_data        : a numpy array of data (modulated)
        carrier_frequency : an integer freqency of the carrier signal
    Returns:
        demodulated_data  : a numpy array of data (demodulated)
    """
    samples = len(modulated_data)
    duration = samples / sample_rate
    time_array = np.linspace(0, duration, samples, False)
    carrier_signal_sin = np.sin(carrier_frequency * time_array * 2 * np.pi)
    carrier_signal_cos = np.cos(carrier_frequency * time_array * 2 * np.pi)
    # demodulate by dividing data by carrier
    demodulated_data_sin = np.divide(modulated_data, carrier_signal_sin)
    demodulated_data_cos = np.divide(modulated_data, carrier_signal_cos)
    # Find average of signal
    average_sin = np.average(demodulated_data_sin)
    average_cos = np.average(demodulated_data_cos)
    # DC Block
    demodulated_data_sin = demodulated_data_sin - average_sin
    demodulated_data_cos = demodulated_data_cos - average_cos
    # Make complex
    demodulated_data = np.array()
    for i in range(len(samples)):
        demodulated_data[i] = complex(demodulated_data_sin[i], demodulated_data_cos[i])
    return demodulated_data
