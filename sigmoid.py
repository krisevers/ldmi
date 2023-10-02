import numpy as np
import pylab as plt

import IPython

"""
Implementation of sigmoid function.
"""

def sigmoid(x, R=1):
    return 1 / (1 + np.exp(-R * x))

x = np.linspace(-1, 1, 1000)

R = np.linspace(0.1, 10, 100)

R = [2/3]

colors = plt.cm.Spectral(np.linspace(0, 1, len(R)))
plt.figure()
for i, r in enumerate(R):
    plt.plot(x, sigmoid(x, r), color=colors[i])
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid function')
plt.show()
