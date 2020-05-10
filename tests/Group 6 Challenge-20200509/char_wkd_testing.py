from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, array, sqrt, log2, vectorize, cos, sin, pi, fft
from utilities import bitarray2dec, dec2bitarray, signal_power

#1) Remove channel response
#signal convoluted with channel response
#received_signal= 

CP = 32
num_subcarriers = 1024
K= num_subcarriers
allCarriers = np.arange(K)

mapping_table = {
    (0,0) :  1+1j,
    (0,1) : -1+1j,
    (1,0) : -1-1j,
    (1,1) :  1-1j,
}
for b1 in [0, 1]:
    for b0 in [0, 1]:
        B = (b1, b0)
        Q = mapping_table[B]
        #plt.plot(Q.real, Q.imag, 'bo')
        #plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
        #plt.show()
demapping_table = {v : k for k, v in mapping_table.items()}




channel_response = np.array([7.3962975e-01, 1.6636183e-01, -7.2079662e-02,
3.4944028e-01, 1.9209278e-01,  -3.1723910e-01,  -2.2716427e-01,   6.1162001e-02,
-5.3678952e-02,  -1.3797263e-01,   8.1535854e-02,   1.6035785e-01,   3.1053201e-03,
-2.7264990e-02,   5.4091047e-02,   1.0230260e-01,   1.5276981e-01,  -1.0168228e-01,
-3.1122081e-02,   2.5037851e-02,  -4.9399505e-03,   1.5189808e-02,   2.6092513e-02,
-4.8524226e-03,  -8.4381426e-03,   7.1524222e-03,  -3.8939963e-03,  -1.5183588e-02,
-1.3195137e-03, 7.4107627e-03])



H_exact = np.fft.fft(channel_response, K)
plt.plot(allCarriers, abs(H_exact))
plt.show()


#2) Remove channel response

#???

#3) Remove cyclic prefix and take the DFT to transform back

def ofdm_rx(received_signal, nfft, nsc, CP):
    """ OFDM Receive Signal Processing """

    num_ofdm_symbols = int(len(received_signal) / (nfft + CP))
    x_hat = zeros([nsc, num_ofdm_symbols], dtype=complex)

    for i in range(0, num_ofdm_symbols):
        ofdm_symbol = y[i * nfft + (i + 1) * cp_length:(i + 1) * (nfft + CP)]
        symbols_freq = fft(ofdm_symbol)
        x_hat[:, i] = concatenate((symbols_freq[-nsc / 2:], symbols_freq[1:(nsc / 2) + 1]))

    return x_hat



#3) Demap

def Demapping(QPSK):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])

    index_list = abs(input_symbols - self._constellation[:, None]).argmin(0)
    demod_bits = dec2bitarray(index_list, self.num_bits_symbol)


return demod_bits






