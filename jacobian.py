import numpy as np

def sigmoid(x, a, b, d):
    return (a*x-b) / (1 + np.exp(-d * (a*x-b)))

def dsigmoid(x, a, b, d):
    return (a * d * np.exp(-d * (a*x-b))) / (1 + np.exp(-d * (a*x-b)))**2

def dXdt(X, V, theta):
    tau_s = theta['tau_s']
    tau_m = theta['tau_m']
    R = theta['tau_m'] / theta['C_m']
    G = theta['G']

    a = theta['a']
    b = theta['b']
    d = theta['d']

    dX = (-X + np.dot(G, sigmoid(V, a, b, d)))/tau_m
    dV = (R*X - V)/tau_s

    return dX, dV

def jacobian(X, V, theta):
    tau_s = theta['tau_s']
    tau_m = theta['tau_m']
    R = theta['tau_m'] / theta['C_m']
    G = theta['G']

    a = theta['a']
    b = theta['b']
    d = theta['d']

    dXX = -1/tau_s
    dVX = R/tau_m
    dXV = -G * dsigmoid(V, a, b, d) / tau_m
    dVV = 0

    return np.array([[dXX, dXV], [dVX, dVV]])

if __name__=="__main__":
    theta = {
            'tau_s':    .5e-3,
            'tau_m':    10e-3,
            'C_m':      250e-6,
            'a':        48,
            'b':        981,
            'd':        8.9e-3,
            'G':        0.1009
            }
    
    X = np.linspace(0, .1, 100)
    V = np.linspace(0, .1, 100)

    XX, VV = np.meshgrid(X, V)

    dX, dV = dXdt(XX, VV, theta)

    import pylab as plt

    plt.figure(figsize=(5, 10))
    plt.quiver(XX, VV, dX, dV)
    plt.xlabel('X')
    plt.ylabel('V')
    plt.show()