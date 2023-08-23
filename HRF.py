import numpy as np
import scipy as sp
import pylab as plt

def HRF(t):
    return 5.7*t**5 * np.exp(-t) / sp.special.factorial(6) - 0.95*t**15 * np.exp(-t) / sp.special.factorial(16)

if __name__=='__main__':

    t = np.linspace(0, 30, 1000)
    
    plt.figure()
    plt.plot(t, HRF(t))
    plt.show()
