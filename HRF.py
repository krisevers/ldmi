import numpy as np
import scipy as sp
import pylab as plt

from scipy.stats import gamma

def HRF(times, shape=[1, 1, 1], scale=[1, 1, 1], loc=[0, 0, 0]):
    """ Return values for HRF at given times """
    num_modes = len(shape)
    values = np.zeros_like(times)
    
    for i in range(num_modes):
        values += gamma.pdf(times, shape[i], loc=loc[i]) * scale[i]

    return values

if __name__=='__main__':

    t = np.linspace(0, 30, 1000)

    U = np.zeros((10000, 1))
    U[1000:5000] = 1

    # create sets of random shapes scales and locations
    shapes = [6, 3, 10]
    scales = [1, -0.1, -0.2]
    locs   = [0, 0, 0]

    hrf = HRF(t, shapes, scales, locs)

    plt.figure()
    plt.plot(t, hrf)
    plt.show()

    # convolve all hrfs with stimulus
    X = np.convolve(U[:, 0], hrf, mode='same')

    plt.figure()
    plt.plot(X, alpha=1)
    plt.show()

    import IPython; IPython.embed()