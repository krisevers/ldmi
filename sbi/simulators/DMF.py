import numpy as np
import copy

class DMF:
    def __init__(self):
        self.name = 'DMF'

        # --------------------------------------------------------------------------
        # Default dynamics parameters
        self.P = {}

        self.P['dt'] = 1e-4

        self.P['sigma'] = 0.0
        self.P['tau_s'] = 0.5e-3
        self.P['tau_m'] = 10e-3
        self.P['C_m']   = 250e-6
        self.P['R']     = self.P['tau_m'] / self.P['C_m']

        self.P['nu_bg'] = 8
        self.P['K_bg']  = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])

        self.P['a'] = 48
        self.P['b'] = 981
        self.P['d'] = 8.9e-3

        # --------------------------------------------------------------------------
        # Default structural parameters
        M = 8
        self.P['M'] = M

        g = -4
        self.P['J_E'] = 87.8e-3
        self.P['J_I'] = self.P['J_E'] * g

        self.P['W'] = np.tile([self.P['J_E'], self.P['J_I']], (M, int(M/2)))

        self.P['P'] = np.array(
                        [[0.1009, 0.1689, 0.0440, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
                         [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
                         [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
                         [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
                         [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
                         [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
                         [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                         [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
        self.P['P'][0, 2] *= 2   # double the connection from L4E to L23E

        self.P['N'] = np.array([20683,  5834,   21915,  5479,   4850,   1065,   14395,  2948  ])

        self.P['W_bg'] = self.P['K_bg'] * self.P['J_E']
        
    def sim(self, E):

        M = self.P['M']  # number of populations

        # Set up connectivity matrix
        self.P['K_rec'] = np.log(1-self.P['P']) / np.log(1 - 1/(self.P['N'] * self.P['N'])) / self.P['N']
        self.P['W'] *= self.P['K_rec']

        # --------------------------------------------------------------------------
        # Initial condtions:
        I = np.zeros(self.P['M'])
        H = np.zeros(self.P['M'])
        F = np.zeros(self.P['M'])

        dt = self.P['dt']
        X = np.zeros((int(E['T'] / dt), self.P['M']))

        a = self.P['a']
        b = self.P['b']
        d = self.P['d']

        def f(h, a, b, d):
            # gain function
            h = np.float128(h)
            return (a * h - b) / (1 - np.exp(-d * (a * h - b)))

        for t in range(int(E['T'] / dt)):

            I += dt * (-I / self.P['tau_s'])
            I += dt * np.dot(self.P['W'], F)
            I += dt * (self.P['W_bg'] * self.P['nu_bg'])
            I += dt * E['U'][t, :]
            I += np.sqrt(dt/self.P['tau_s']) * self.P['sigma'] * np.random.randn(M)
            H += dt * ((-H + self.P['R']*I) / self.P['tau_m'])
            F = f(H, a=self.P['a'], b=self.P['b'], d=self.P['d'])

            X[t, :] = I.T   # save synaptic activity for computing BOLD signal

        return X    # synaptic activity