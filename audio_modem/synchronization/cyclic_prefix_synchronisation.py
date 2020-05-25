import numpy as np



def cp_synchronisation(input_data,ofdm_block_length,CP,threshold=0.5):
    
    
    N=ofdm_block_length
    PREFIXED_SYMBOL_LENGTH = N+CP
    CONSTELLATION_VALUES_PER_BLOCK = int((N-2)/2)
    #Normalise input data and change into a list of floats
    data = input_data
    data = np.array(data).astype(np.float32)
    data *= 1.0 / np.max(np.abs(data))
    data = data.tolist()

    # Correlate
    delayed_data = [0] * N + data[:-N]
    prod = [datum * delayed_datum for datum, delayed_datum in zip(data, delayed_data)]

    # Accumulate
    acc = [0]
    for datum in prod:
        acc.append(acc[-1] + datum)

    # Differentiate
    diff = np.diff(acc)

    # Extremify
    diff = [1 if datum >= 0 else -1 for datum in diff]
    #axs[3].plot(diff)

    # Detect symbols with a moving average window of width CP
    avg = []
    for i in range(len(diff[CP:])):
        avg.append(np.average(diff[i : i + CP]))
    avg = [datum ** 3 for datum in avg]
    # Denoise
    avg = np.array(avg).astype(np.float32)
    avg *= 1.0 / np.max(np.abs(avg))
    avg = avg.tolist()
    
    # Detect most common locations of cyclic prefix within the symbol
    chunks = [avg[i : i + PREFIXED_SYMBOL_LENGTH] for i in range(0, len(avg), PREFIXED_SYMBOL_LENGTH)]
    chunks[-1] += [0] * (PREFIXED_SYMBOL_LENGTH - len(chunks[-1]))
    scores = [0] * PREFIXED_SYMBOL_LENGTH
    
    for i in range(len(scores)):
        for chunk in chunks:
            if chunk[i] >= threshold:
                scores[i] += 1
                
    max_score= max(scores)
    shifts = []
    for i in range(len(scores)):
        if scores[i] == max_score:
            shifts.append(i)
            
    shifts = [shift + CP for shift in shifts]

    # For each possible shift value, retrieve the first OFDM symbol
    deviations = {}
    for shift in shifts:
        # Shift data to synchronise
        shifted_data = data[shift:]
        shifted_data = [shifted_data[i : i + PREFIXED_SYMBOL_LENGTH] for i in range(0, len(shifted_data), PREFIXED_SYMBOL_LENGTH)]

        # Remove all data blocks whose power is less than the normalised cutoff power
        power_list = [np.sqrt(np.mean(np.square(block))) for block in shifted_data]
        power_list = np.array(power_list)
        power_list = power_list - np.min(power_list)
        power_list *= 1.0 / np.max(power_list)
        power_list = power_list.tolist()
        cutoff = 0.5
        power_list = [0 if datum < cutoff else 1 for datum in power_list]
        shifted_data = [shifted_data[i] for i in range(len(shifted_data)) if power_list[i] == 1]

        # Extract first symbol and remove cyclic prefix
        shifted_data = shifted_data[0][CP:]

        # FFT and extract encoded data
        shifted_data = np.fft.fft(shifted_data, n=N)
        shifted_data = shifted_data[1 : 1 + CONSTELLATION_VALUES_PER_BLOCK]

        # Check arguments of first quadrant
        # To check if it's a circle or a cluster
        shifted_data = [np.arctan(datum.imag / datum.real) for datum in shifted_data]
        shifted_data = [datum for datum in shifted_data if datum >= 0 and datum <= np.pi / 2]
        deviations[np.std(shifted_data)] = shift

    shift = deviations[min(deviations.keys())]

    return shift