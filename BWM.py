import numpy as np

import numpy as np

def BWM(theta):

    t_sim = 50
    dt = 0.001

    T = int(t_sim/dt)

    # P-DCM
    U = np.zeros((int(t_sim/dt)))
    U[int(10/dt):int(15/dt)] = 1

    X = np.zeros((T, 1))
    F = np.zeros((T, 1))

    xinflow = 0
    xvaso   = 0
    yinflow = 0
    yvaso   = 0

    for t in range(T):
        X[t] = X[t] + dt * (-X[t] + U[t])      # neural activity

        xinflow = np.exp(xinflow)
        yvaso   = yvaso + dt * (X[t] - theta['c1'] * xvaso)   # vasoactive signal
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

        q = q + q_dot
        v = v + v_dot

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

    return U, F, V, Q, BOLD

if __name__ == '__main__':

    import pylab as plt

    theta = {'c1': 0.6, 'c2': 1.5, 'c3': 0.6,
             'tau_mtt': 2, 'tau_vs': 4, 'alpha': 0.32, 'E_0': 0.4, 'V_0': 4, 'eps': 0.0463, 'rho_0': .191, 'nu_0': 126.3, 'TE': 0.028}
    U, F, V, Q, BOLD = BWM(theta)

    plt.figure(figsize=(5, 10))
    plt.subplot(5, 1, 1)
    plt.title('Stimulus')
    plt.plot(U, lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, U.shape[0])
    plt.subplot(5, 1, 2)
    plt.title('Cerebral Blood Flow')
    plt.plot(F, lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, F.shape[0])
    plt.subplot(5, 1, 3)
    plt.title('Cerebral Blood Volume')
    plt.plot(V, lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, V.shape[0])
    plt.subplot(5, 1, 4)
    plt.title('Deoxyhemoglobin Content')
    plt.plot(Q, lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, Q.shape[0])
    plt.subplot(5, 1, 5)
    plt.title('BOLD Signal')
    plt.plot(BOLD, lw=3, c='black')
    plt.xlim(0, BOLD.shape[0])
    plt.xticks(np.linspace(0, BOLD.shape[0], 6), np.round(np.linspace(6, 50, 6), 0))
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()