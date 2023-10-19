import numpy as np

def smoothness(x):
    """
    Calculates the smoothness data x
    """
    return np.std(np.diff(x))/abs(np.mean(np.diff(x)))

if __name__=="__main__":

    import pylab as plt

    x_uni = np.random.uniform(-2, 2, 100)
    x_norm = np.random.normal(0,  1, 100)
    x_sin  = np.sin(np.linspace(0, 2*np.pi, 100))
    x_sin_noise = x_sin + np.random.normal(0, 0.1, 100)

    y_uni = smoothness(x_uni)
    y_norm = smoothness(x_norm)
    y_sin = smoothness(x_sin)
    y_sin_noise = smoothness(x_sin_noise)

    plt.figure()
    plt.plot(x_uni)
    plt.plot(x_norm)
    plt.plot(x_sin)
    plt.plot(x_sin_noise)
    plt.show()

    print(y_uni)
    print(y_norm)
    print(y_sin)
    print(y_sin_noise)