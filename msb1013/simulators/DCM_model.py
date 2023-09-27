import numpy as np
import copy

def DCM_sim(U, P):

    K = P['K']  # number of depths
    M = P['M']  # number of populations

    # Neuronal parameters:
    # --------------------------------------------------------------------------
    C = P['C']  # external connection

    # Initial condtions:
    Xn = np.zeros(M)
    yn = np.zeros(M)

    dt = P['dt']
    neuro = np.zeros((int(P['T'] / dt), M))

    for t in range(int(P['T'] / dt)):

        yn = yn + dt * (np.dot(P['W'], Xn) + np.dot(C, U['u'][t, :].T))

        Xn = copy.deepcopy(yn)

        neuro[t, :] = yn.T

    return neuro

def DCM_parameters(K, P):
    P['K'] = K      # number of depths
    P['M'] = 2*K    # number of populations

    # if K < 10:
    #     P['dt'] = 0.01  # default integration step
    # elif K < 20:
    #     P['dt'] = 0.005  # smaller for higher number of cortical depths
    # else:
    #     P['dt'] = 0.001

    P['dt'] = 0.001

    # Neuronal parameter:
    # --------------------------------------------------------------------------
    sigma = -3
    mu = -1.5
    lambda_ = 0.2
    P['C'] = np.eye(K*2)

    W = np.array([[sigma,    mu],
                  [lambda_, -lambda_]])
    Z = np.zeros((2, 2))
    P['W'] = np.asarray(np.bmat([[W, Z, Z, Z],
                                 [Z, W, Z, Z],
                                 [Z, Z, W, Z],
                                 [Z, Z, Z, W]]))

    return P