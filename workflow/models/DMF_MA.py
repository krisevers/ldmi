import numpy as np
import json

from numba import jit

"""
Implement multi-area model which takes external input and interarea connectivity as the parameters and returns the steady state currents of the populations.
"""

def DMF_MA(I_ext, A, areas=['V1', 'hMT'], num_columns=[4, 2]):
    """
    Multi-area DMF: Dynamic Mean Field

    Takes external input and interarea connectivity as the parameters and returns the steady state currents of the populations.
    
    Parameters
    ----------
    I_ext : array
        External input currents
    A : array
        Interarea connectivity
    areas : list
        List of areas
    """

    # load population size
    with open('maps/popsize.json') as f:
        popsizes = json.load(f)

    # neuronal parameters
    tau_m = 10e-3
    tau_s = .5e-3
    C_m   = 250e-6
    R     = tau_m / C_m

    a = 48.
    b = 981.
    d = 8.9e-3

    # connectivity parameters
    P = np.array( # connection probabilities
                [[0.1009, 0.1689, 0.0440, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
                 [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
                 [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
                 [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
                 [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
                 [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
                 [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                 [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
    

    
    # load population size
    with open('maps/popsize.json') as f:
        popsizes = json.load(f)
    N = 
    for area in areas:
        N = np.array(popsizes[area]) / 2

if __name__=="__main__":

    import pylab as plt