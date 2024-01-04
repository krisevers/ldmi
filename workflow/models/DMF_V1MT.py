import numpy as np
import json

from numba import jit

"""
Implement multi-area model which takes external input and interarea connectivity as the parameters and returns the steady state currents of the populations.
"""

def DMF_MA(I_ext=None, A=None, G=1, areas=['V1', 'MT'], num_columns=[2, 2], sigma=0):
    """
    Multi-area DMF: Dynamic Mean Field

    Takes external input and interarea connectivity as the parameters and returns the steady state currents of the populations.
    
    Parameters
    ----------
    I_ext : array
        External input currents
    A : array
        Interarea connectivity
    G : float
        Global coupling strength
    areas : list
        List of areas
    """

    t_sim = 1
    dt = 1e-4
    T = int(t_sim / dt)

    # neuronal parameters
    tau_m = 10e-3
    tau_s = .5e-3
    C_m   = 250e-6
    R     = tau_m / C_m

    a = 48.
    b = 981.
    d = 8.9e-3

    # connectivity parameters
    P = np.array( # connection probabilities
                [[0.1009, 0.1689, 0.0440, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
                 [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
                 [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
                 [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
                 [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
                 [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
                 [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                 [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])

    tot_num_columns = np.sum(num_columns)   # total number of columns
    num_areas = len(areas)                  # number of areas
    M = 8                                   # number of populations per area
    tot_num_pops = tot_num_columns * M      # total number of populations

    Z       = np.zeros((M, M))    # zeros array to fill off-diagonal blocks of P_all
    P_all   = np.zeros((tot_num_pops, tot_num_pops))   # interarea connectivity matrix

    # fill diagonal blocks of P_all
    for i in range(tot_num_columns):
        P_all[i*M:(i+1)*M, i*M:(i+1)*M] = P

    if A is None:
        P = P_all       # no connections between columns
    else:
        P = P_all + A   # total connectivity matrix
    
    # load population size
    with open('maps/popsize.json') as f:
        popsizes = json.load(f)
    N = []
    for area in areas:
        N_ = np.array(popsizes[area]) / 2
        N_ = np.tile(N_, (num_columns[areas.index(area)], 1))
        N.append(N_)
    N = np.concatenate(N).ravel()

    C = np.log(1-P) / np.log(1 - 1/(N * N)) / N     # number of synapses

    g = -8.
    J_E = 87.8e-3
    J_I = J_E * g

    G = np.tile([J_E, J_I], (tot_num_pops, int(tot_num_pops/2))) * C * G   # synaptic conductances



    C_bg = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])   # number of background synapses
    G_bg = C_bg * J_E
    nu_bg = 8.
    I_bg = G_bg * nu_bg
    I_bg = np.tile(I_bg, tot_num_columns)

    # estimate stability of the system 
    # eigenvalues of the connectivity matrix
    eigvals = np.linalg.eigvals(G)
    # eigenvalues of the connectivity matrix + background input
    eigvals_bg = np.linalg.eigvals(G + np.diag(I_bg))

    eigs = np.concatenate((eigvals, eigvals_bg))


    if I_ext is None:
        I_ext = np.zeros((T, tot_num_pops))
    else:
        I_ = np.zeros((T, tot_num_pops))
        I_[int(0.2/dt):int(0.8/dt), :] = I_ext
        I_ext = I_


    I = np.zeros((T, tot_num_columns, M**2+M*2), dtype=np.float32)                   # all currents (recurrent + external)
    X = np.zeros((T, tot_num_pops),                             dtype=np.float32)    # synaptic current
    Y = np.zeros((T, tot_num_pops),                             dtype=np.float32)    # membrane potential

    @jit(nopython=True)
    def func(x, a=a, b=b, d=d):
        return (a*x - b) / (1 - np.exp(-d*(a*x - b)))
    
    dsig = np.sqrt(dt / tau_s) * sigma

    # @jit(nopython=True)
    def sim(X, Y, I, I_ext, I_bg, G, func, tau_s, tau_m, R, dt, T):
        for t in range(1, T):
            # save currents (recurrent + external)
            I_rec_all = G * func(Y[t-1]) * dt
            for i in range(tot_num_columns):
                lower = i*M
                upper = (i+1)*M
                I_rec_loc                   = I_rec_all[lower:upper, lower:upper]
                I_rec_inter                 = I_rec_all[lower:upper, :]
                I_rec_inter[:, lower:upper] = 0
                I[t, i, :M**2]                                              = np.ravel(I_rec_loc)                       * dt    # local recurrent
                I[t, i, M**2:M**2+M]                                        = np.sum(I_ext[t-1, lower:upper], axis=0)   * dt    # thalamocortical
                I[t, i, M**2+M:]                                            = np.sum(I_rec_inter, axis=1)               * dt    # corticocortical

            # update state variables
            X_dot = (-X[t-1]/tau_s + np.dot(G, func(Y[t-1])) + I_ext[t-1] + I_bg)
            Y_dot = (-Y[t-1] + R*X[t-1]) / tau_m

            X[t] = X[t-1] + dt * X_dot + dsig * np.random.randn(tot_num_pops)
            Y[t] = Y[t-1] + dt * Y_dot

        return X, Y, I

    X, Y, I = sim(X=X, Y=Y, I=I, I_ext=I_ext, I_bg=I_bg, G=G, func=func, tau_s=tau_s, tau_m=tau_m, R=R, dt=dt, T=T)

    F = func(Y)

    return X, Y, F, I, eigs

if __name__=="__main__":

    import pylab as plt

    """
    V1 and MT are each modelled by a single column, we explore hypotheses about the connectivity between these two areas.
    """

    areas = ['V1', 'MT']                    # list of areas
    num_areas = len(areas)                  # number of areas
    num_columns = [1, 1]                    # number of columns per area
    tot_num_columns = np.sum(num_columns)   # total number of columns
    M = 8                                   # number of populations per area
    tot_num_pops = tot_num_columns * M      # total number of populations

    mask = np.zeros((tot_num_pops, tot_num_pops), dtype=int)
    # feedforward
    mask[M:M*2, :M]           = 1 # V1 -> MT
    # feedback
    mask[:M, M:M*2]           = 2 # MT -> V1

    conn_types = np.unique(mask)    # 0: no connection, 1/2: within area, 3/4: feedforward, 5/6: feedback
    num_conn_types = len(conn_types)

    A = np.zeros((tot_num_pops, tot_num_pops), dtype=float)

    for i in range(1, num_conn_types):
        x = np.random.rand(M, M)
        x[x < 0.9] = 0  # make x sparse
        # scale between 0 and .35
        x_max = .1
        x[x > 0] = (x[x > 0] - x[x > 0].min()) / (x[x > 0].max() - x[x > 0].min()) * x_max
        # fill x in at the right places where mask == conn_types[i]
        x = np.tile(x, (tot_num_columns, tot_num_columns))
        A[mask == conn_types[i]] = x[mask == conn_types[i]]

    A[:, 1::2] = 0  # set inhibitory connections to 0



    plt.figure()
    plt.imshow(A, cmap='Reds')
    plt.xticks(np.arange(M/2, tot_num_pops, M), [r'$V1$', r'$MT$'])
    plt.yticks(np.arange(M/2, tot_num_pops, M), [r'$V1$', r'$MT$'])
    # horizontal and vertical lines between column blocks
    for i in range(1, tot_num_columns):
        plt.axvline(i*M-0.5, color='k')
        plt.axhline(i*M-0.5, color='k')
    plt.colorbar()
    plt.show()

    I_V1 = np.random.randn(M)
    I_MT = np.random.randn(M)
    I_ext = np.concatenate((I_V1, I_MT))
    I_ext[I_ext < 0.75] = 0
    I_ext *= 1e-3

    X, Y, F, I, eigs = DMF_MA(areas=areas, num_columns=num_columns, A=A, I_ext=None, G=1, sigma=0.02)

    # plt.figure()
    # # V1
    # plt.subplot(2, 1, 1)
    # plt.imshow(X[:, :M], cmap='Reds', aspect='auto', interpolation='none')
    # plt.xticks([])
    # plt.yticks([])
    # plt.ylabel('Time (ms)')
    # plt.xlabel('Population')
    # plt.title(r'$V1$')
    # # MT
    # plt.subplot(2, 1, 2)
    # plt.imshow(X[:, M:M*2], cmap='Reds', aspect='auto', interpolation='none')
    # plt.xticks([])
    # plt.yticks([])
    # plt.ylabel('Time (ms)')
    # plt.xlabel('Population')
    # plt.title(r'$MT$')
    # plt.tight_layout()
    # plt.show()


    colors = plt.cm.Spectral(np.linspace(0, 1, M))
    plt.figure()
    for i in range(M):
        plt.subplot(2, 1, 1)
        plt.title(r'$V1$')
        plt.plot(F[:, i])
        plt.subplot(2, 1, 2)
        plt.title(r'$MT$')
        plt.plot(F[:, M+i])
    plt.tight_layout()
    plt.show()

























    # import sys
    # # add parent directory to path
    # sys.path.append('maps/')

    # from I2K import I2K

    # K = 5
    # T = I.shape[0]
    # MAP = np.zeros((T, tot_num_columns, K))
    # all_areas = np.repeat(areas, num_columns)
    # for i, area in enumerate(all_areas):
    #     PROB_K = I2K(K, 'macaque', area, sigma=1)

    #     # flatten probabilities along last two dimensions
    #     PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])

    #     MAP[:, i] = (I[:, i] @ PROB_K.T)

    # import seaborn as sns
    # sns.set_style('white')

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.imshow(MAP[:, 0], cmap='Reds', aspect='auto', interpolation='none')
    # plt.xticks([])
    # plt.yticks([])
    # plt.ylabel('Time (ms)')
    # plt.xlabel('Depth')
    # plt.title(r'$V1$')
    # plt.subplot(2, 1, 2)
    # plt.imshow(MAP[:, 1], cmap='Reds', aspect='auto', interpolation='none')
    # plt.xticks([])
    # plt.yticks([])
    # plt.ylabel('Time (ms)')
    # plt.xlabel('Depth')
    # plt.title(r'$MT$')
    # plt.tight_layout()
    # plt.show()

    # plt.figure()
    # dt = 1e-4
    # plt.plot(MAP[int(0.5/dt), 0] - MAP[int(0.1/dt), 0], label=r'$V1$', color='k')
    # plt.plot(MAP[int(0.5/dt), 1] - MAP[int(0.1/dt), 1], label=r'$MT$', color='r')
    # plt.legend()
    # plt.xlabel('Depth')
    # plt.ylabel('Current')
    # plt.show()
    