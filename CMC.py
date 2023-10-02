import numpy as np
import pylab as plt

"""
Implementation of the canonical microcircuit (CMC) model of the neocortex.
"""

tau = [256, 128, 16, 32]

G = np.array([[-8, -4, -4,  0],
              [ 4, -8, -2,  0],
              [ 4,  2, -4,  2],
              [ 0,  1, -2, -4]])

# states
P = np.zeros(4)
Q = np.zeros(4)
R = np.zeros(4)

def f(X, dt, T):
    """implement CMC differential equations (from Wei et al. (2020))"""
    global P, Q, R
    P = X[:4]
    Q = X[4:8]
    R = X[8:12]

    dP = np.zeros(4)
    dQ = np.zeros(4)
    dR = np.zeros(4)

    for i in range(4):
        dP[i] = (1/T[i]) * (-P[i] + np.sum(G[i,:] * R))
        dQ[i] = (1/T[i]) * (-Q[i] + np.sum(G[i,:] * P))
        dR[i] = (1/T[i]) * (-R[i] + np.sum(G[i,:] * Q))

    return np.concatenate((dP, dQ, dR))

def simulate(X0, dt, T, tau):
    """simulate CMC model"""
    X = np.zeros((int(T/dt), 12))
    X[0,:] = X0
    for i in range(1, int(T/dt)):
        X[i,:] = X[i-1,:] + dt * f(X[i-1,:], dt, tau)
    return X

def plot(X, dt, T):
    """plot CMC model"""
    t = np.arange(0, T, dt)
    plt.figure()
    plt.plot(t, X[:,0], label='P_L23E')
    plt.plot(t, X[:,1], label='P_L23I')
    plt.plot(t, X[:,2], label='P_L4E')
    plt.plot(t, X[:,3], label='P_L4I')
    plt.plot(t, X[:,4], label='Q_L23E')
    plt.plot(t, X[:,5], label='Q_L23I')
    plt.plot(t, X[:,6], label='Q_L4E')
    plt.plot(t, X[:,7], label='Q_L4I')
    plt.plot(t, X[:,8], label='R_L23E')
    plt.plot(t, X[:,9], label='R_L23I')
    plt.plot(t, X[:,10], label='R_L4E')
    plt.plot(t, X[:,11], label='R_L4I')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    dt = 0.01
    T = 1000
    X0 = np.zeros(12)
    X0[0] = 1
    X0[1] = 1
    X0[2] = 1
    X0[3] = 1

    X = simulate(X0, dt, T, tau)
    plot(X, dt, T)
