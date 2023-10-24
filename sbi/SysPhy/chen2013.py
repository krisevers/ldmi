import numpy as np
import pylab as plt

from worker import F

J_E = 87.8e-3

theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,                                     # intrinsic neuronal parameters
            'I_L4E': 0.0983*902*18*J_E, 'I_L4I': 0.0619*902*18*J_E, 'I_L6E': 0.0512*902*20*J_E, 'I_L6I': 0.0196*902*20*J_E, # external input
            'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,                                                        # neurovascular coupling parameters
            'E_0v': 0.35, 'V_0t': 2, 'TE': 0.028}                                                                           # hemodynamic parameters

E = {'K': 22, 'area': 'V1', 'T': 50, 'onset': 10, 'offset': 20}   # experimental parameters

Psi, X, S, F_l, F_k, B_k, B_v = F(E, theta, test=True)  # forward model

# casting to float32
X = X.astype(np.float32)
S = S.astype(np.float32)
F_l = F_l.astype(np.float32)
F_k = F_k.astype(np.float32)
B_k = B_k.astype(np.float32)
B_v = B_v.astype(np.float32)

Chen2013_X = np.array([
                0.40954222493225806,
                0.4522709412773616, 
                0.5854525415825667, 
                0.6300105321851186, 
                0.4614955702868471, 
                0.4238112162880406, 
                0.41772667838358674,
                0.29562506725273
            ])

Chen2013_Y = np.array([
            -0.00790402994662176,
                0.37221329142197523,
                0.5643311084228133,
                0.7450624268371813,
                0.9309245171661573,
                1.137348824014686,
                1.3112423739480041,
                1.4969968599088948,
            ]) * int(E['K']/1.5)


plt.figure(figsize=(7, 7))
peaks = np.max(B_k, axis=0)
plt.plot(peaks, np.arange(0, E['K']), linewidth=2, color='black', label='Model')
plt.plot(Chen2013_X, Chen2013_Y, linewidth=2, color='grey', linestyle='--', label='Chen et al. (2013)')
plt.xlabel('Signal Change (%)', fontsize=20)
plt.ylabel('Cortical Depth', fontsize=20)
plt.legend(loc='upper right', fontsize=16)
plt.xlim(0, 1)
plt.ylim(0, E['K'])
plt.gca().invert_yaxis()
plt.savefig('pdf/ff_chen2013.pdf', format='pdf', dpi=1200)

plt.show()

