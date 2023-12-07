import numpy as np

from numba import jit

@jit(nopython=True)
def NVC(neuro, c1=0.6, c2=1.5, c3=0.6):
    # --------------------------------------------------------------------------

    T = neuro.shape[0]
    K = neuro.shape[1]

    # Initial condtions:
    Xvaso    = np.zeros(K)
    Yvaso    = np.zeros(K)
    Xinflow  = np.zeros(K)
    Yinflow  = np.zeros(K)

    dt = 1e-4
    cbf = np.zeros((T, K))

    for t in range(T):
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