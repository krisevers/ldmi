import numpy as np

def balloon_windkessel(neural_activity, dt=0.001, 
                       tau=0.98, alpha=0.32, 
                       E_0=0.34, V_0=0.02, 
                       k1=3, k2=1, k3=1, 
                       tau_s=2, tau_f=1.5):
    """
    Simulates the BOLD response using the Balloon-Windkessel model.
    
    :param neural_activity: array-like, the neural activity over time
    :param dt: float, the time step
    :param tau: float, the hemodynamic transit time
    :param alpha: float, the Grubb's exponent
    :param E_0: float, the resting oxygen extraction fraction
    :param V_0: float, the resting blood volume fraction
    :param k1, k2, k3: floats, the BOLD signal coefficients
    :param tau_s: float, the time constant for the signal decay
    :param tau_f: float, the time constant for the signal rise
    
    :return: array-like, the BOLD response over time
    """
    n_time_points = len(neural_activity)
    s = np.zeros(n_time_points, dtype=np.float128)
    f = np.zeros(n_time_points, dtype=np.float128)
    v = np.ones(n_time_points, dtype=np.float128) * V_0
    q = np.ones(n_time_points, dtype=np.float128) * V_0 * E_0
    bold = np.zeros(n_time_points, dtype=np.float128)
    
    for t in range(1, n_time_points):
        ds = dt * (-s[t-1]/tau_s + neural_activity[t-1])                                                # neural activity
        df = dt * (s[t-1] - f[t-1])/tau_f                                                               # vasodilatory signal
        dv = dt * ((f[t-1] - v[t-1]**(1/alpha))/tau)                                                    # blood volume
        dq = dt * ((f[t-1] * (1 - (1 - E_0)**(1/f[t-1])) - (v[t-1]**(1 - alpha) * q[t-1])/v[t-1])/tau)  # deoxyhemoglobin
        
        s[t] = s[t-1] + ds
        f[t] = f[t-1] + df
        v[t] = v[t-1] + dv
        q[t] = q[t-1] + dq
        
        bold[t] = v[t] * (k1 + k2) - (k2 / k3) * q[t]
        
    return bold

if __name__ == '__main__':

    # Example usage:
    import pylab as plt

    t_sim = 60
    dt = 0.001

    T = int(t_sim/dt)

    U = np.zeros((int(t_sim/dt), 2))
    U[int(10/dt):int(20/dt), 0] = 1

    _sigma   = -3
    _mu      = -1.5
    _lambda_ = 0.2
    W = np.array([[_sigma,    _mu     ],           # within layer connectivity
                  [_lambda_, -_lambda_]])
    
    x = np.zeros(2)
    X = np.zeros((T, 2))
    for t in range(T):
        x_dot = dt * (np.dot(W, x) + U[t])
        x = x + x_dot
        X[t] = x


    # neurovascular coupling (NVC)
    c1 = 0.6
    c2 = 1.5
    c3 = 0.6


    # Get the BOLD response
    bold_response = balloon_windkessel(X[:, 0])

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(X, label='Neural Activity')
    plt.plot(bold_response, label='BOLD Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
