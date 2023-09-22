import numpy as np
import copy

class DCM:
    def __init__(self, E):
        self.name = 'DCM'

        # --------------------------------------------------------------------------
        # Default parameters
        self.P = {}

        self.P['K'] = E['K']  # number of depths
        self.P['M'] = 2 * E['K']  # number of populations

        self.P['dt'] = 0.001  # default integration step

        # Default structural parameters:
        # --------------------------------------------------------------------------
        sigma = -3
        mu = -1.5
        lambda_ = 0.2
        self.P['C'] = np.eye(self.P['K'] * 2)

        W = np.array([[sigma,    mu     ],
                      [lambda_, -lambda_]])
        Z = np.zeros((2, 2))
        self.P['W'] = np.asarray(np.bmat([[W, Z, Z, Z],
                                          [Z, W, Z, Z],
                                          [Z, Z, W, Z],
                                          [Z, Z, Z, W]]))

def sim(self, E):

    K = self.P['K']  # number of depths
    M = self.P['M']  # number of populations

    # Neuronal parameters:
    # --------------------------------------------------------------------------
    C = self.P['C']  # external connection

    # Initial condtions:
    Xn = np.zeros(M)
    yn = np.zeros(M)

    dt = self.P['dt']
    neuro = np.zeros((int(E['T'] / dt), M))

    for t in range(int(E['T'] / dt)):

        yn = yn + dt * (np.dot(self.P['W'], Xn) + np.dot(C, E['U'][t, :].T))

        Xn = copy.deepcopy(yn)

        neuro[t, :] = yn.T

    return neuro
