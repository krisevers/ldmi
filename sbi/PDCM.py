import numpy as np

from simulators.NVC import NVC

def worker(theta):

    t_sim = 70
    dt = 0.001

    T = int(t_sim/dt)

    E = {'K': 1, 'T': t_sim, 'dt': dt}

    # P-DCM
    U = np.zeros((int(t_sim/dt), 2))
    U[int(10/dt):int(40/dt), 0] = 1

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
        E_f = 1 - (1 - E_0)**(1 / F[t])

        v_dot = dt * ((F[t] - f_out) / tau_mtt)
        q_dot = dt * ((F[t] * (E_f / E_0) - f_out * (q / v)) / tau_mtt)

        f_out = v ** (1 / alpha) + tau_vs * v_dot

        q = q + q_dot
        v = v + v_dot

        V[t] = v
        Q[t] = q

        k1 = 4.3 * nu_0  * E_0 * TE
        k2 = eps * rho_0 * E_0 * TE
        k3 = 1 - eps

        BOLD[t] = V_0 * (k1*(1 - Q[t]) + k2*(1 - Q[t]/V[t]) + k3*(1 - V[t]))

    # remove initial transient up to stimulus onset
    U = U[int(10/dt):]
    X = X[int(10/dt):]
    F = F[int(10/dt):]
    V = V[int(10/dt):]
    Q = Q[int(10/dt):]
    BOLD = BOLD[int(10/dt):]

    stats = gen_stats(BOLD[:,0], dt)

    return U, X, F, V, Q, BOLD, stats


def gen_stats(BOLD, dt):

    if np.any(np.isnan(BOLD)):
        return {
            'mean': np.nan,
            'std': np.nan,
            'skew': np.nan,
            'kurt': np.nan,
            'max_val': np.nan,
            'max_pos': np.nan,
            'max_slope': np.nan,
            'min_slope': np.nan,
            'max_slope_pos': np.nan,
            'min_slope_pos': np.nan,
            'positive_area': np.nan,
            'negative_area': np.nan,
            'ratio_area': np.nan,
        }
    
    else:

        # get moments of BOLD signal
        mean = np.mean(BOLD)
        std = np.std(BOLD)
        skew = np.mean((BOLD - mean)**3) / std**3
        kurt = np.mean((BOLD - mean)**4) / std**4

        # find max values
        max_val = np.max(BOLD)
        max_pos = np.argmax(BOLD)

        # time to minimum after peak
        min_pos_as = np.argmin(BOLD[max_pos:])
        tta = (min_pos_as - max_pos) * dt

        # time to mimimum before peak
        min_pos_bs = np.argmin(BOLD[:max_pos])
        ttb = (max_pos - min_pos_bs) * dt

        max_pos     = max_pos * dt
        min_pos_as  = min_pos_as * dt
        min_pos_bs  = min_pos_bs * dt 

        # find max and min slopes
        dBOLD = np.diff(BOLD)
        max_slope = np.max(dBOLD)
        min_slope = np.min(dBOLD)

        max_slope_pos = np.argmax(dBOLD) * dt 
        min_slope_pos = np.argmin(dBOLD) * dt

        # find positive area under BOLD response
        positive_area = np.sum(BOLD[BOLD > 0]) * dt

        # find negative area under BOLD response
        negative_area = np.sum(BOLD[BOLD < 0]) * dt

        ratio_area = positive_area / (negative_area + positive_area)

        # find peak-to-peak amplitude
        tts = max_slope_pos - min_slope_pos

        stats = {
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurt': kurt,
            'max_val': max_val,
            'max_pos': max_pos,
            'max_slope': max_slope,
            'min_slope': min_slope,
            'max_slope_pos': max_slope_pos,
            'min_slope_pos': min_slope_pos,
            'positive_area': positive_area,
            'negative_area': negative_area,
            'ratio_area': ratio_area,
            'tts': tts,
            'tta': tta,
            'ttb': ttb,
        }

        return stats
            
if __name__ == '__main__':

    import pylab as plt

    theta = {'c1': 0.6, 'c2': 1.5, 'c3': 0.6,
             'tau_mtt': 2, 'tau_vs': 4, 'alpha': 0.32, 'E_0': 0.4, 'V_0': 2, 'eps': 0.2, 'rho_0': .191, 'nu_0': 126.3, 'TE': 0.028}
    U, X, F, V, Q, BOLD, stats = worker(theta)

    plt.figure(figsize=(5, 10))
    plt.subplot(6, 1, 1)
    plt.title('Stimulus')
    plt.plot(U[:, 0], lw=2, c='black')
    plt.xticks([])
    plt.xlim(0, U.shape[0])
    plt.subplot(6, 1, 2)
    plt.title('Neuronal Response')
    plt.plot(X[:, 0], lw=2, c='black')
    plt.xticks([])
    plt.xlim(0, X.shape[0])
    plt.subplot(6, 1, 3)
    plt.title('Cerebral Blood Flow')
    plt.plot(F, lw=2, c='black')
    plt.xticks([])
    plt.xlim(0, F.shape[0])
    plt.subplot(6, 1, 4)
    plt.title('Cerebral Blood Volume')
    plt.plot(V, lw=2, c='black')
    plt.xticks([])
    plt.xlim(0, V.shape[0])
    plt.subplot(6, 1, 5)
    plt.title('Deoxyhemoglobin Content')
    plt.plot(Q, lw=2, c='black')
    plt.xticks([])
    plt.xlim(0, Q.shape[0])
    plt.subplot(6, 1, 6)
    plt.title('BOLD Signal')
    plt.plot(BOLD, lw=2, c='black')
    plt.xlim(0, BOLD.shape[0])
    plt.xticks(np.linspace(0, BOLD.shape[0], 6), np.round(np.linspace(0, 40, 6), 0))
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('pdf/PDCM_timeseries.pdf')
    plt.show()

    print(np.array(stats.values()))
    print(np.array(theta.values()))

    import IPython; IPython.embed()