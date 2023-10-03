import numpy as np

from simulators.NVC import NVC

def worker(theta):

    t_sim = 50
    dt = 0.001

    T = int(t_sim/dt)

    E = {'K': 1, 'T': t_sim, 'dt': dt}

    # P-DCM
    U = np.zeros((int(t_sim/dt), 2))
    U[int(10/dt):int(15/dt), 0] = 1

    _sigma   = -3
    _mu      = -1.5
    _lambda_ = 0.2
    W = np.array([[_sigma,    _mu     ],      
                  [_lambda_, -_lambda_]])

    x = np.zeros(2)
    X = np.zeros((T, 2))
    for t in range(T):
        x_dot = dt * (np.dot(W, x) + U[t])
        x = x + x_dot
        X[t] = x

    # neurovascular coupling (NVC)
    NVC_model = NVC()
    NVC_model.P['c1'] = theta['c1']
    NVC_model.P['c2'] = theta['c2']
    NVC_model.P['c3'] = theta['c3']
    NVC_model.P['dt'] = dt
    F = NVC_model.sim(X[:, 0], E)

    # hemodynamic model (balloon model Buxton et al., 1998)
    v     = 0    # blood volume
    q     = 0    # deoxyhemoglobin content
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
        E_f = 1 - (1 - E_0)**(1 / F[t])

        v_dot = dt * ((F[t] - f_out) / tau_mtt)
        v = v + v_dot

        q_dot = dt * ((F[t] * (E_f / E_0) - f_out * (q / v))/ tau_mtt)
        q = q + q_dot

        f_out = v ** (1 / alpha) + tau_vs * v_dot

        V[t] = v
        Q[t] = q

        k1 = 4.3 * nu_0  * E_0 * TE
        k2 = eps * rho_0 * E_0 * TE
        k3 = 1 - eps

        BOLD[t] = V_0 * (k1*(1 - Q[t]) + k2*(1 - Q[t]/V[t]) + k3*(1 - V[t]))

    # remove initial transient
    U = U[int(6/dt):]
    X = X[int(6/dt):]
    F = F[int(6/dt):]
    V = V[int(6/dt):]
    Q = Q[int(6/dt):]
    BOLD = BOLD[6000:]

    if np.any(np.isnan(BOLD)):
        return U, X, F, V, Q, BOLD, np.nans(6)

    else:
        # compute summary statistics from BOLD signal (e.g. peak and undershoot)
        # peak location
        peak_idx = np.argmax(BOLD)
        peak_time = peak_idx * dt - 10 + 6
        peak_amp  = BOLD[peak_idx, 0]
        # undershoot location
        if peak_idx == len(BOLD) - 1:
            undershoot_idx = peak_idx
            undershoot_time = peak_time
            undershoot_amp  = peak_amp
        else:
            undershoot_idx = np.argmin(BOLD[peak_idx:]) + peak_idx
            undershoot_time = undershoot_idx * dt - 10 + 6
            undershoot_amp  = BOLD[undershoot_idx, 0]
        # initial dip location
        if peak_idx == 0:
            initial_dip_idx = peak_idx
            initial_dip_time = peak_time
            initial_dip_amp  = peak_amp
        else:
            initial_dip_idx = np.argmin(BOLD[:peak_idx])
            initial_dip_time = initial_dip_idx * dt - 10 + 6
            initial_dip_amp  = BOLD[initial_dip_idx, 0]

        if peak_time > initial_dip_time and peak_time < undershoot_time and peak_amp > initial_dip_amp and peak_amp > undershoot_amp:

            stats = np.array([peak_time, peak_amp, undershoot_time, undershoot_amp, initial_dip_time, initial_dip_amp])

            return U, X, F, V, Q, BOLD, stats
        
        else: 

            stats = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            return U, X, F, V, Q, BOLD, stats

if __name__ == '__main__':

    import pylab as plt

    theta = {'c1': 0.6, 'c2': 1.5, 'c3': 0.6,
             'tau_mtt': 2, 'tau_vs': 4, 'alpha': 0.32, 'E_0': 0.4, 'V_0': 4, 'eps': 0.0463, 'rho_0': .191, 'nu_0': 126.3, 'TE': 0.028}
    U, X, F, V, Q, BOLD, stats = worker(theta)

    plt.figure(figsize=(5, 10))
    plt.subplot(6, 1, 1)
    plt.title('Stimulus')
    plt.plot(U[:, 0], lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, U.shape[0])
    plt.subplot(6, 1, 2)
    plt.title('Neuronal Response')
    plt.plot(X[:, 0], lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, X.shape[0])
    plt.subplot(6, 1, 3)
    plt.title('Cerebral Blood Flow')
    plt.plot(F, lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, F.shape[0])
    plt.subplot(6, 1, 4)
    plt.title('Cerebral Blood Volume')
    plt.plot(V, lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, V.shape[0])
    plt.subplot(6, 1, 5)
    plt.title('Deoxyhemoglobin Content')
    plt.plot(Q, lw=3, c='black')
    plt.xticks([])
    plt.xlim(0, Q.shape[0])
    plt.subplot(6, 1, 6)
    plt.title('BOLD Signal')
    plt.plot(BOLD, lw=3, c='black')
    plt.xlim(0, BOLD.shape[0])
    plt.xticks(np.linspace(0, BOLD.shape[0], 6), np.round(np.linspace(6, 50, 6), 0))
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('svg/PDCM_timeseries.svg')
    plt.show()

    plt.figure()
    colors = ['black', 'red', 'blue']
    plt.subplot(1, 2, 1)
    plt.title('Feature Timing')
    plt.bar(np.arange(3), stats[::2], color=colors)
    plt.xticks(np.arange(3), ['peak time', 'undershoot time', 'initial dip time'],
               rotation=45, ha='right')
    plt.subplot(1, 2, 2)
    plt.title('Feature Amplitude')
    plt.bar(np.arange(3), stats[1::2], color=colors)
    plt.xticks(np.arange(3), ['peak amplitude', 'undershoot amplitude', 'initial dip amplitude'],
               rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('svg/PDCM_stats.svg')
    plt.show()

    print(np.array(stats))
    print(np.array(theta.values()))

    import IPython; IPython.embed()