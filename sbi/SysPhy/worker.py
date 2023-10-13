import numpy as np

from scipy import ndimage
from scipy import signal

from utils import get_L2K, get_N
from LBR import LBR

def F(E, theta={}, test=False):
    """
    Black-Box forward model for the DMF, NVC and LBR models

    E - experiment parameters (dict)
    theta - parameters (dict)
    
    Returns:
    Psi - observables (dict)
    """

    ##########################################################################################
    # Experimental parameters
    dt_DMF = 1e-4
    T = int(E['T'] / dt_DMF)  # [ms]
    M = 8       # number of populations
    L = 4       # number of layers

    ##########################################################################################
    # Dynamic Mean Field (DMF)
    P = np.array(                                                                   # connection probabilities
                [[0.1009, 0.1689, 0.0440, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
                 [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
                 [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
                 [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
                 [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
                 [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
                 [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                 [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
    # update connection probabilities (according the Potjans & Diesmann (2014); Fig. 11)
    if 'P_L23EtoL23E' in theta:         # superficial self-excitation
        P[0, 0] = theta['P_L23EtoL23E']
    if 'P_L4EtoL23E' in theta:          # feedforward granular to superficial excitation
        P[0, 2] = theta['P_L4EtoL23E']
    if 'P_L23EtoL5E' in theta:          # feedforward deep to superficial excitation
        P[4, 0] = theta['P_L23EtoL5E']
    if 'P_L23EtoL4I' in theta:          # feedback superficial to granular inhibition
        P[3, 0] = theta['P_L23EtoL4I']
    if 'P_L23EtoL6I' in theta:          # feedback superficial to deep inhibition
        P[7, 0] = theta['P_L23EtoL6I']
    if 'P_L5EtoL23I' in theta:          # feedback deep to superficial inhibition
        P[1, 4] = theta['P_L5EtoL23I']
    if 'P_L6E_L4I' in theta:            # feedback deep to granular inhibition
        P[3, 6] = theta['P_L6E_L4I']
    if 'P_L5E_L6E' in theta:            # feedforward deep to deep excitation
        P[6, 4] = theta['P_L5E_L6E']
    
    # N = get_N(E['area'])                                                # number of neurons in each population
    N = np.array( [20683,  5834,   21915,  5479,   4850,   1065,   14395,  2948  ])

    C = np.log(1-P) / np.log(1 - 1/(N * N)) / N                         # number of synapses

    if 'J_E' in theta:
        J_E = theta['J_E']                                              # excitatory synaptic weight
    else:
        J_E = 87.8e-3
    if 'J_I' in theta:
        J_I = theta['J_I']                                              # inhibitory synaptic weight
    else:
        J_I = J_E * -4

    G = np.tile([J_E, J_I], (M, int(M/2))) * C                          # synaptic weights matrix

    # background input
    C_bg = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])   # number of background synapses
    W_bg = C_bg * J_E                                                   # background synaptic weight
    nu_bg = 8                                                           # background firing rate

    X = np.zeros((T, M))    # synaptic current
    Y = np.zeros((T, M))    # membrane potential

    # external input
    U = np.zeros((T, M))    # external input
    onset   = int(E['onset'] /dt_DMF)
    offset  = int(E['offset']/dt_DMF)
    I_ext = np.zeros(M)
    if 'I_L23E' in theta:
        I_ext[0] = theta['I_L23E']
    if 'I_L23I' in theta:
        I_ext[1] = theta['I_L23I']
    if 'I_L4E' in theta:
        I_ext[2] = theta['I_L4E']
    if 'I_L4I' in theta:
        I_ext[3] = theta['I_L4I']
    if 'I_L5E' in theta:
        I_ext[4] = theta['I_L5E']
    if 'I_L5I' in theta:
        I_ext[5] = theta['I_L5I']
    if 'I_L6E' in theta:
        I_ext[6] = theta['I_L6E']
    if 'I_L6I' in theta:
        I_ext[7] = theta['I_L6I']
    [U[onset:offset, i].fill(I_ext[i]) for i in range(M)]

    # intrinsic neuronal parameters
    if 'tau_m' in theta:
        tau_m = theta['tau_m']
    else:
        tau_m = 10e-3
    if 'tau_s' in theta:
        tau_s = theta['tau_s']
    else:
        tau_s = .5e-3
    if 'C_m' in theta:
        C_m   = theta['C_m']
    else:
        C_m   = 250e-6
    R     = tau_m / C_m

    if 'a' in theta:
        a = theta['a']
    else:
        a = 48
    if 'b' in theta:
        b = theta['b']
    else:
        b = 981
    if 'd' in theta:
        d = theta['d']
    else:
        d = 8.9e-3

    def func(x):
        return (a*x-b) / (1 + np.exp(-d * (a*x-b)))

    for t in range(1, T):
        X_dot = (-X[t-1]/tau_s + np.dot(G, func(Y[t-1])) + U[t-1] + W_bg*nu_bg)
        Y_dot = (-Y[t-1] + R*X[t-1]) / tau_m

        X[t] = X[t-1] + dt_DMF * X_dot
        Y[t] = Y[t-1] + dt_DMF * Y_dot

    # neural signal from excitatory and inhibitory populations
    if 'lam_E' in theta:
        lam_E = theta['lam_E']
    else:
        lam_E = 1
    if 'lam_I' in theta:
        lam_I = theta['lam_I']
    else:
        lam_I = 0

    # remove initial transient (first second)
    X_base = X[int(1/dt_DMF)]   # baseline synaptic current
    X = X[int(1/dt_DMF):]
    Y = Y[int(1/dt_DMF):]

    T = X.shape[0]

    S = lam_E*abs(X[:,::2] - X_base[::2]) + lam_I*abs(X[:,1::2] - X_base[1::2])    # neural signal

    ##########################################################################################
    # Neurovascular Coupling (NVC)
    if 'c1' in theta:
        c1 = theta['c1']
    else:
        c1 = 0.6
    if 'c2' in theta:
        c2 = theta['c2']
    else:
        c2 = 1.5
    if 'c3' in theta:
        c3 = theta['c3']
    else:
        c3 = 0.6

    dt_NVC = 1e-4

    a_x = np.zeros(L)
    a_y = np.zeros(L)
    f_x = np.zeros(L)
    f_y = np.zeros(L)

    F_l = np.zeros((T, L))

    for t in range(T):
        f_x = np.exp(f_x)
        a_dot = S[t] - c1*a_x
        f_dot = c2*a_x - c3*(f_x-1)

        a_y = a_y + dt_NVC * a_dot
        f_y = f_y + dt_NVC * (f_dot / f_x)

        a_x = a_y 
        f_x = f_y

        F_l[t] = np.exp(f_y)        # cerebral blood flow at each cortical layer


    ##########################################################################################
    # Layer to Depth Mapping (L2K)
    K       = E['K']       # number of cortical depths
    area    = E['area']    # cortical area

    L2K, _ = get_L2K(K, L, area)

    F_k = np.zeros((T, K))    # blood flow at each cortical depth
    for t in range(T):
        F_k[t] = np.sum(L2K*F_l[t], axis=1)

    # smooth blood flow across cortical depths
    F_k = ndimage.gaussian_filter1d(F_k, 1, axis=1)

    ##########################################################################################
    # Laminar BOLD Response (LBR)
    dt_LBR = 0.001

    F_k = F_k[::int(dt_LBR/dt_NVC)]    # downsample to match LBR sampling rate

    lbr_model = LBR(K, theta)
    
    B_k, _, _ = lbr_model.sim(F_k, K)    # BOLD signal at each cortical depth

    # downsample BOLD signal to match voxel space
    num_voxels = 3
    n = int(K/num_voxels)
    window = np.ones((1, n))/n
    B_v = signal.convolve2d(B_k, window, mode='valid')[:,::n]

    ##########################################################################################
    # Observables
    Psi = {}    # observables

    Psi['peak_pos'] = np.argmax(B_v, axis=0)     # peak position
    Psi['peak_amp'] = np.max(B_v, axis=0)        # peak amplitude
    Psi['area']     = np.trapz(B_v, axis=0)      # area under the curve

    Psi['upslope']  = np.zeros(num_voxels)       # upslope
    Psi['downslope']= np.zeros(num_voxels)       # downslope
    for i in range(num_voxels):
        Psi['upslope'][i]   = np.max(np.diff(B_v[:,i]))
        Psi['downslope'][i] = np.min(np.diff(B_v[:,i]))

    # difference between voxels
    Psi['peak_dpos']   = np.diff(Psi['peak_pos'])
    Psi['peak_damp']   = np.diff(Psi['peak_amp'])
    Psi['area_d']      = np.diff(Psi['area'])
    Psi['upslope_d']   = np.diff(Psi['upslope'])
    Psi['downslope_d'] = np.diff(Psi['downslope'])

    if test:
        return Psi, X, S, F_l, F_k, B_k, B_v

    else:
        return Psi





if __name__=="__main__":

    import pylab as plt

    J_E = 87.8e-3

    theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,                             # intrinsic neuronal parameters
             'I_L4E': 0.0983*902*15*J_E, 'I_L4I': 0.0619*902*15*J_E,                                                    # external input
             'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,                                                   # neurovascular coupling parameters
             'E_0v': 0.35, 'V_0t': 2, 'TE': 0.028}                                                                      # hemodynamic parameters
    
    E = {'K': 12, 'area': 'V1', 'T': 50, 'onset': 10, 'offset': 20}   # experimental parameters

    Psi, X, S, F_l, F_k, B_k, B_v = F(E, theta, test=True)  # forward model

    # casting to float32
    X = X.astype(np.float32)
    S = S.astype(np.float32)
    F_l = F_l.astype(np.float32)
    F_k = F_k.astype(np.float32)
    B_k = B_k.astype(np.float32)
    B_v = B_v.astype(np.float32)

    fig = plt.figure(figsize=(7, 7))
    plt.subplot(5, 1, 1)
    plt.title(r'Layer specific neural activity ($S$)')
    plt.imshow(S.T, aspect='auto', cmap='Reds', interpolation='none')
    plt.yticks(np.arange(4), ['L23', 'L4', 'L5', 'L6'])
    plt.colorbar()
    plt.subplot(5, 1, 2)
    plt.title(r'Cerebral Blood Flow before upsampling ($F_l$)')
    plt.imshow(F_l.T, aspect='auto', cmap='Reds', interpolation='none')
    plt.yticks(np.arange(4), ['L23', 'L4', 'L5', 'L6'])
    plt.colorbar()
    plt.subplot(5, 1, 3)
    plt.title(r'Cerebral Blood Flow after upsampling ($F_k$)')
    plt.imshow(F_k.T, aspect='auto', cmap='Reds', interpolation='none')
    plt.colorbar()
    plt.subplot(5, 1, 4)
    plt.title(r'BOLD signal before downsampling ($B_k$)')
    plt.imshow(B_k.T, aspect='auto', cmap='Reds', interpolation='none')
    fig.text(0.07, 0.4, 'Cortical depth (K)', va='center', rotation='vertical')
    plt.colorbar()
    plt.subplot(5, 1, 5)
    plt.title(r'BOLD signal after downsampling ($B_v$)')
    plt.imshow(B_v.T, aspect='auto', cmap='Reds', interpolation='none')
    plt.yticks(np.arange(3), ['Superficial', 'Granular', 'Deep'])
    plt.colorbar()
    plt.tight_layout(pad=1)
    plt.savefig('pdf/ff_L4.pdf', format='pdf', dpi=1200)

    # plot Psi observables
    plt.figure(figsize=(7, 7))
    plt.subplot(5, 1, 1)
    plt.title(r'Peak position ($\Psi_{peak\_pos}$)')
    plt.plot(Psi['peak_pos'])
    plt.subplot(5, 1, 2)
    plt.title(r'Peak amplitude ($\Psi_{peak\_amp}$)')
    plt.plot(Psi['peak_amp'])
    plt.subplot(5, 1, 3)
    plt.title(r'Area under the curve ($\Psi_{area}$)')
    plt.plot(Psi['area'])
    plt.subplot(5, 1, 4)
    plt.title(r'Upslope ($\Psi_{upslope}$)')
    plt.plot(Psi['upslope'])
    plt.subplot(5, 1, 5)
    plt.title(r'Downslope ($\Psi_{downslope}$)')
    plt.plot(Psi['downslope'])
    plt.tight_layout(pad=1)
    plt.savefig('pdf/ff_L4_Psi.pdf', format='pdf', dpi=1200)

    plt.figure(figsize=(7, 7))
    peaks_S = np.max(S, axis=0)
    plt.barh(width=peaks_S, y=np.arange(4), linewidth=2, color='black')
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('pdf/ff_L4_S.pdf', format='pdf', dpi=1200)

    plt.figure(figsize=(7, 7))
    peaks_F_k = np.max(F_k, axis=0)
    plt.plot(peaks_F_k, np.arange(1, E['K']+1), linewidth=2, color='black')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig('pdf/ff_L4_F_k.pdf', format='pdf', dpi=1200)

    plt.figure(figsize=(7, 7))
    peaks = np.max(B_k, axis=0)
    plt.plot(peaks, np.arange(1, E['K']+1), linewidth=2, color='black')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig('pdf/ff_L4_B_k.pdf', format='pdf', dpi=1200)


    plt.show()
    

    import IPython; IPython.embed()
