import numpy as np

def NVC_sim(neuro, P):

    K = P['K']

    # NVC parameters:
    # --------------------------------------------------------------------------
    c1 = P['c1']
    c2 = P['c2']
    c3 = P['c3']

    # Initial condtions:
    Xvaso    = np.zeros(K)
    Yvaso    = np.zeros(K)
    Xinflow  = np.zeros(K)
    Yinflow  = np.zeros(K)

    dt = P['dt']
    cbf = np.zeros((int(P['T'] / dt), K))

    for t in range(int(P['T'] / dt)):
        Xinflow = np.exp(Xinflow)
        # ----------------------------------------------------------------------
        # Vasoactive signal:
        Yvaso = Yvaso + dt * (neuro[t] - c1 * Xvaso)
        # ----------------------------------------------------------------------
        # Inflow:
        df_a = c2 * Xvaso - c3 * (Xinflow - 1)
        Yinflow = Yinflow + dt * (df_a / Xinflow)

        Xvaso   = np.copy(Yvaso)
        Xinflow = np.copy(Yinflow)

        cbf[t, :] = np.exp(Yinflow).T

    return cbf

def NVC_parameters(K, P):

    P['K'] = K

    # if K < 10:
    #     P['dt'] = 0.01  # default integration step
    # elif K < 20:
    #     P['dt'] = 0.005  # smaller for higher number of cortical depths
    # else:
    #     P['dt'] = 0.001

    P['dt'] = 0.001

    # NVC parameters:
    # --------------------------------------------------------------------------
    P['c1'] = 0.6
    P['c2'] = 1.5
    P['c3'] = 0.6

    return P