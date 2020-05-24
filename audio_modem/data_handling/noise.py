import numpy as np

def add_noise_amp(input_data, amplitude):
    scale = max(input_data)
    noise = scale * amplitude * np.random.normal(0, 1, len(input_data))
    return [datum + noise_datum for datum, noise_datum in zip(input_data, noise)]


def add_noise_db(input_data, SNR=1000):
    # Preprocess
    data = input_data
    data = np.array(data)
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()
    print(data[1])

    # Add AGWN
    SNR = (10) ** (SNR / 20)
    noise_magnitude = 1 / SNR
    noise = noise_magnitude * np.random.normal(0, 1, len(data))
    noise = noise.tolist()
    return [datum + noise_datum for datum, noise_datum in zip(data, noise)]
