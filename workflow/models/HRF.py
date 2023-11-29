import numpy as np

"""
Hemodynamic response function.
"""

def HRF(t, a1=6, a2=6, b1=0.9, b2=0.9, c=0.35, d=0.1, e=0.5):
    """
    Hemodynamic response function. two gamma functions with positive and negative
    """
    pos_gamma = (t / b1)**a1 * np.exp(-(t - b1) / c)
    neg_gamma = d * (t / b2)**a2 * np.exp(-(t - b2) / e)

    return (pos_gamma - neg_gamma) / np.max(pos_gamma - neg_gamma)

if __name__=="__main__":

    import pylab as plt

    # explore parameter space
    dt = 0.001
    t = np.arange(0, 10, dt)

    plt.figure()
    plt.plot(t, HRF(t))


    timesteps = np.arange(0, 100, dt)
    protocol = np.zeros((len(timesteps)))
    protocol[int(10/dt):int(20/dt)] = 1

    X = np.convolve(protocol, HRF(t))[:len(timesteps)]
    X /= np.max(X)

    plt.figure()
    plt.plot(protocol)
    plt.plot(X)
    plt.show()


    import IPython; IPython.embed();