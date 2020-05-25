from itertools import product
from numpy import arange, array, sqrt, log2, vectorize, cos, sin, pi
from utilities import bitarray2dec, dec2bitarray, signal_power

# Equate "psk_form" to the form of PSK used (4,16 etc)

psk_form= 16

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




class PSKModem(Modem):
    """ m : int- Size of the PSK constellation.
        constellation : 1D-ndarray of complex- Modem constellation. 
        Es            : float- Average energy per symbols.
        m             : integer- Constellation length.
        num_bits_symb : integer- Number of bits per symbol.
        """

    def __init__(self, m):
        def _constellation_symbol(i):
            return cos(2 * pi * (i - 1) / m) + sin(2 * pi * (i - 1) / m) * (0 + 1j)

        self.constellation = list(map(_constellation_symbol, arange(m)))




psk = PSKModem(psk_form)
symbols= psk.modulate([1,0,0,1])
symbols

