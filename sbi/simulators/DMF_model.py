import numpy as np
import copy

def DMF_sim(U, P):

    M = P['M']  # number of populations

    # Neuronal parameters:
    # --------------------------------------------------------------------------
    # Initial condtions:
    I = np.zeros(M)
    H = np.zeros(M)
    F = np.zeros(M)

    dt = P['dt']
    neuro = np.zeros((int(P['T'] / dt), M))

    def f(h, a, b, d):
        # gain function
        h = np.float128(h)
        return (a * h - b) / (1 - np.exp(-d * (a * h - b)))

    for t in range(int(P['T'] / dt)):

        I += dt * (-I / P['tau_s'])
        I += dt * np.dot(P['W'], F)
        I += dt * (P['W_bg'] * P['nu_bg'])
        I += dt * U[t, :]
        I += np.sqrt(dt/P['tau_s']) * P['sigma'] * np.random.randn(M)
        H += dt * ((-H + P['R']*I) / P['tau_m'])
        F = f(H, a=P['a'], b=P['b'], d=P['d'])

        neuro[t, :] = I.T   # save synaptic input for computing BOLD signal

    return neuro

def DMF_parameters(P):
    P['dt'] = 1e-4

    P['sigma'] = 0.0
    P['tau_s'] = 0.5e-3
    P['tau_m'] = 10e-3
    P['C_m']   = 250e-6
    P['R']     = P['tau_m'] / P['C_m']

    P['nu_bg'] = 8
    P['K_bg']  = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])

    M = 8
    P['M'] = M

    g = -4
    P['J_E'] = 87.8e-3
    P['J_I'] = P['J_E'] * g

    P['W'] = np.tile([P['J_E'], P['J_I']], (M, int(M/2)))

    P['P'] = np.array(
				     [[0.1009, 0.1689, 0.0440, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
				      [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
				      [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
				      [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
				      [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
				      [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
				      [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
				      [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
    # P['P'][0, 2] *= 2   # double the connection from L4E to L23E

    P['N'] = np.array([20683,  5834,   21915,  5479,   4850,   1065,   14395,  2948  ])

    P['W_bg'] = P['K_bg'] * P['J_E']

    P['a'] = 48
    P['b'] = 981
    P['d'] = 8.9e-3

    return P


# TODO: add DMF model equations and PD14 parameters
# TODO: add proper N2K transformation with the right cortical depth profiles for the layers of the model