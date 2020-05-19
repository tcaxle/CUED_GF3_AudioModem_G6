import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

x = [complex(0, 0)] * 100
x[1] = complex(1, -1) / np.sqrt(2)
x[-1] = np.conj(x[1])
#x[50] = complex(1, 0)
y = np.fft.ifft(x)

plt.plot(y.real)
plt.plot(y.imag)
plt.show()
