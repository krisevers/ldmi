import numpy as np
import json

from numba import jit

def DMF(I_th, I_cc, area=None, N=None, t_sim=1, dt=1e-4, sigma=0):
    """
    DMF: Dynamic Mean Field

    Takes external input currents and returns the steady state currents
    of the populations.
    """

    T = int(t_sim / dt)
    M = 8

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
    P[0, 2] *= 2
    
    # load population size
    if N is None:
        with open('maps/popsize.json') as f:
            popsizes = json.load(f)
        N = np.array(popsizes[area]) / 2
        if area == 'V1':
            N = np.array([20683,	5834,	21915,	5479,	4850,	1065,	14395,	2948])

    C = np.log(1-P) / np.log(1 - 1/(N * N)) / N     # number of synapses

    g = -4.
    J_E = 87.8e-3
    J_I = J_E * g

    G = np.tile([J_E, J_I], (M, int(M/2))) * C    # synaptic conductances

    C_bg = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])   # number of background synapses
    G_bg = C_bg * J_E
    nu_bg = 8.
    I_bg = G_bg * nu_bg

    I = np.zeros((T, M*M + M*2), dtype=np.float32)    # all currents (recurrent + thalamic and cortico-cortical)
    X = np.zeros((T, M),         dtype=np.float32)    # synaptic current
    Y = np.zeros((T, M),         dtype=np.float32)    # membrane potential

    @jit(nopython=True)
    def func(x, a=a, b=b, d=d):
        return (a*x - b) / (1 - np.exp(-d*(a*x - b)))
    
    @jit(nopython=True)
    def sim(X, Y, I, I_th, I_cc, I_bg, G, func, tau_s, tau_m, R, dt, T, N, sigma=0):
        for t in range(1, T):
            dsig = np.sqrt(dt / tau_s) * sigma
            # save currents (recurrent + external)
            I[t, :M**2]         = np.ravel(G * func(Y[t-1]))  * dt  #* np.repeat(N, M) * dt
            I[t, M**2:M**2+M]   = I_th[t-1]                   * dt  #* N * dt
            I[t, M**2+M:]       = I_cc[t-1]                   * dt  #* N * dt
            # update state variables
            X_dot = (-X[t-1]/tau_s + np.dot(G, func(Y[t-1])) + I_th[t-1] + I_cc[t-1] + I_bg)
            Y_dot = (-Y[t-1] + R*X[t-1]) / tau_m

            X[t] = X[t-1] + dt * X_dot + dsig * np.random.randn(M)
            Y[t] = Y[t-1] + dt * Y_dot

        return X, Y, I
    
    X, Y, I = sim(X=X, Y=Y, I=I, I_th=I_th, I_cc=I_cc, I_bg=I_bg, G=G, func=func, tau_s=tau_s, tau_m=tau_m, R=R, dt=dt, T=T, N=N, sigma=sigma)

    F = np.zeros((T, M), dtype=np.float32)
    for t in range(1, T):
        F[t] = func(Y[t])
    
    return X, Y, I, F