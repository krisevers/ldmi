import numpy as np
import pylab as plt



def sigm(x, a=1, b=0, d=1):
    return (a*x-b) / (1 + np.exp(-d * (a*x-b)))

def f(x, y, G, U, tau=.002):
    x_dot = -x/tau + y + U
    y_dot = -(x - y)/tau

    return x_dot, y_dot

if __name__=="__main__":

    t_sim = 1
    dt = 0.001
    T = int(t_sim/dt)

    X = np.zeros(T)
    Y = np.zeros(T)
    x = 0
    y = 0

    G = np.array([1, 0])
    U = np.zeros(T)
    U[int(0.2/dt):int(0.4/dt)] = 1

    for t in range(T):
        x_dot, y_dot = f(x, y, G, U[t])
        x = x + x_dot * dt
        y = y + y_dot * dt
        X[t] = x
        Y[t] = y

    plt.figure()
    plt.plot(X)
    plt.plot(Y)
    plt.show()