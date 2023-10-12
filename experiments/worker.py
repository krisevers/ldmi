import numpy as np

import numpy as np

def worker(theta):

    t_sim = 50
    dt = 0.001

    T = int(t_sim/dt)

    # P-DCM
    U = np.zeros((int(t_sim/dt), 2))
    U[int(10/dt):int(15/dt), 0] = theta['nu_E']
    U[int(10/dt):int(15/dt), 1] = theta['nu_I']
    C = np.array([1, 0])

    x = np.zeros(2)
    y = np.zeros(2)
    X = np.zeros((T, 2))
    F = np.zeros((T, 1))

    xinflow = 0
    xvaso   = 0
    yinflow = 0
    yvaso   = 0

    def sigm(x, a=theta['a'], b=theta['b'], d=theta['d']):
        return (a*x-b) / (1 + np.exp(-d * (a*x-b)))
    
    G = np.array([[theta['G_EE'], theta['G_IE']], [theta['G_EI'], theta['G_II']]]) * 10
    tau_m = theta['tau_m']
    tau_s = theta['tau_s']
    C_m = theta['C_m']
    R = tau_m / C_m

    for t in range(T):
        x_dot = (-x + np.dot(G, sigm(y)) + C*U[t]) / tau_m
        y_dot = (-y + R*sigm(x)) / tau_s

        x = x + dt * x_dot
        y = y + dt * y_dot

        X[t] = x

        s = x[0]*theta['lam_E'] + abs(x[1]*theta['lam_I'])

        xinflow = np.exp(xinflow)
        yvaso   = yvaso + dt * (s - theta['c1'] * xvaso)   # vasoactive signal
        df_a    = theta['c2'] * xvaso - theta['c3'] * (xinflow - 1)    # inflow
        yinflow = yinflow + dt * (df_a / xinflow)
        xvaso   = yvaso
        xinflow = yinflow

        F[t] = np.exp(yinflow)

    # hemodynamic model (balloon model Buxton et al., 1998 and some modifications according to Havlicek et al., 2015)
    v     = 1    # blood volume
    q     = 1    # deoxyhemoglobin content
    E_f   = 0    # oxygen extraction fraction
    f_out = 0    # outflow of blood

    tau_mtt = theta['tau_mtt']
    tau_vs  = theta['tau_vs']
    alpha   = theta['alpha']
    E_0     = theta['E_0']
    V_0     = theta['V_0']

    eps    = theta['eps']
    rho_0  = theta['rho_0']
    nu_0   = theta['nu_0']
    TE     = theta['TE']

    V = np.zeros((T, 1))
    Q = np.zeros((T, 1))

    BOLD = np.zeros((T, 1))

    for t in range(T):
        E_f = 1 - (1 - E_0)**(1 / F[t])                                     # oxygen extraction fraction

        v_dot = dt * ((F[t] - f_out) / tau_mtt)                             # blood volume
        q_dot = dt * ((F[t] * (E_f / E_0) - f_out * (q / v)) / tau_mtt)     # deoxyhemoglobin content

        f_out = v ** (1 / alpha) + tau_vs * v_dot                           # outflow of blood

        v = v + v_dot
        q = q + q_dot

        V[t] = v
        Q[t] = q

        k1 = 4.3 * nu_0  * E_0 * TE
        k2 = eps * rho_0 * E_0 * TE
        k3 = 1 - eps

        BOLD[t] = V_0 * (k1*(1 - Q[t]) + k2*(1 - Q[t]/V[t]) + k3*(1 - V[t]))    # BOLD signal

    # remove initial transient
    U = U[int(6/dt):]
    F = F[int(6/dt):]
    V = V[int(6/dt):]
    Q = Q[int(6/dt):]
    BOLD = BOLD[6000:]

    return U, X, F, V, Q, BOLD

if __name__ == '__main__':

    import pylab as plt

    theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'nu_E': 1, 'nu_I': 1.5, 'G_EE': 0.1009, 'G_EI': 0.1346, 'G_IE': -0.1689, 'G_II': -0.1371, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,
             'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,
             'tau_mtt': 2, 'tau_vs': 4, 'alpha': 0.32, 'E_0': 0.4, 'V_0': 4, 'eps': 0.0463, 'rho_0': .191, 'nu_0': 126.3, 'TE': 0.028}
    U, X, F, V, Q, BOLD = worker(theta)

    

    plt.figure(figsize=(5, 10))
    plt.subplot(6, 1, 1)
    plt.title('Stimulus')
    plt.plot(U[:,0], lw=1, c='blue', label='Excitatory')
    plt.plot(U[:,1], lw=1, c='red', label='Inhibitory')
    plt.legend(loc='upper right')
    plt.xticks([])
    plt.xlim(0, U.shape[0])
    plt.subplot(6, 1, 2)
    plt.title('Excitatory Population Activity')
    plt.plot(X[:,0], lw=1, c='blue', label='Excitatory')
    plt.plot(X[:,1], lw=1, c='red', label='Inhibitory')
    plt.legend(loc='upper right')
    plt.xticks([])
    plt.xlim(0, X.shape[0])
    plt.subplot(6, 1, 3)
    plt.title('Cerebral Blood Flow')
    plt.plot(F, lw=1, c='black')
    plt.xticks([])
    plt.xlim(0, F.shape[0])
    plt.subplot(6, 1, 4)
    plt.title('Cerebral Blood Volume')
    plt.plot(V, lw=1, c='black')
    plt.xticks([])
    plt.xlim(0, V.shape[0])
    plt.subplot(6, 1, 5)
    plt.title('Deoxyhemoglobin Content')
    plt.plot(Q, lw=1, c='black')
    plt.xticks([])
    plt.xlim(0, Q.shape[0])
    plt.subplot(6, 1, 6)
    plt.title('BOLD Signal')
    plt.plot(BOLD, lw=1, c='black')
    plt.xlim(0, BOLD.shape[0])
    plt.xticks(np.linspace(0, BOLD.shape[0], 6), np.round(np.linspace(6, 50, 6), 0))
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()