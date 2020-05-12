from itertools import product
from numpy import arange, array, sqrt, log2, vectorize
from utilities import bitarray2dec, dec2bitarray, signal_power

# Equate "qam_form" to the form of QAM used (4,16 etc)

qam_form= 16

class Modem:

    """ constellation : array-like with a length which is a power of 2. 1D-ndarray of complex
        Es            : float - Average energy per symbols.
        m             : integer- Constellation length.
        num_bits_symb : integer- Number of bits per symbol.
        """

    def __init__(self, constellation):
        self.constellation = constellation

    def modulate(self, input_bits):
        """ Modulate (map) an array of bits to constellation symbols.
            input_bits : 1D ndarray of ints
            baseband_symbols : 1D ndarray of complex floats- Modulated complex symbols.
        """
        mapfunc = vectorize(lambda i:
                            self._constellation[bitarray2dec(input_bits[i:i + self.num_bits_symbol])])

        baseband_symbols = mapfunc(arange(0, len(input_bits), self.num_bits_symbol))

        return baseband_symbols

    @property
    def constellation(self):
        """ Constellation of the modem. """
        return self._constellation

    @constellation.setter
    def constellation(self, value):
        # Check value input
        num_bits_symbol = log2(len(value))
        if num_bits_symbol != int(num_bits_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        # Set constellation as an array
        self._constellation = array(value)

        # Update other attributes
        self.Es = signal_power(self.constellation)
        self.m = self._constellation.size
        self.num_bits_symbol = int(num_bits_symbol)




class QAMModem(Modem):

    def __init__(self, m):
        """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.
        m : int- Size of the QAM constellation. MUST BE A MULTIPLE OF 2.
        """

        def _constellation_symbol(i):
            return (2 * i[0] - 1) + (2 * i[1] - 1) * (1j)

        mapping_array = arange(1, sqrt(m) + 1) - (sqrt(m) / 2)
        self.constellation = list(map(_constellation_symbol,
                                      list(product(mapping_array, repeat=2))))


qam = QAMModem(qam_form)
symbols= qam.modulate([1,0,0,1])
symbols

