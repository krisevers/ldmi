import numpy as np

class NVC:
    def __init__(self):
        self.name = 'NVC'

        # --------------------------------------------------------------------------
        # Default parameters
        self.P = {}

        self.P['dt'] = 1e-4

        self.P['c1'] = 0.6
        self.P['c2'] = 1.5
        self.P['c3'] = 0.6

    def sim(self, neuro, E):

        K = E['K']

        # Initial condtions:
        Xvaso    = np.zeros(K)
        Yvaso    = np.zeros(K)
        Xinflow  = np.zeros(K)
        Yinflow  = np.zeros(K)

        dt = self.P['dt']
        cbf = np.zeros((int(E['T'] / dt), K))

        for t in range(int(E['T'] / dt)):
            Xinflow = np.exp(Xinflow)
            # ----------------------------------------------------------------------
            # Vasoactive signal:
            Yvaso = Yvaso + dt * (neuro[t] - self.P['c1'] * Xvaso)
            # ----------------------------------------------------------------------
            # Inflow:
            df_a = self.P['c2'] * Xvaso - self.P['c3'] * (Xinflow - 1)
            Yinflow = Yinflow + dt * (df_a / Xinflow)

            Xvaso   = np.copy(Yvaso)
            Xinflow = np.copy(Yinflow)

            cbf[t, :] = np.exp(Yinflow).T

        return cbf