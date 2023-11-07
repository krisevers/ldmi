import numpy as np
import pylab as plt

from worker import F

J_E = 87.8e-3

theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,                                     # intrinsic neuronal parameters
            'I_L4E': 0.0983*902*18*J_E, 'I_L4I': 0.0619*902*18*J_E, 'I_L6E': 0.0512*902*20*J_E, 'I_L6I': 0.0196*902*20*J_E, # external input
            'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,                                                        # neurovascular coupling parameters
            'E_0v': 0.35, 'V_0t': 2, 'TE': 0.028}                                                                           # hemodynamic parameters

E = {'K': 22, 'area': 'V1', 'T': 50, 'onset': 10, 'offset': 20}   # experimental parameters

Psi, X, S, F_l, F_k, B_k, B_v, Y = F(E, theta, mode='full')  # forward model

# casting to float32
X = X.astype(np.float32)
S = S.astype(np.float32)
F_l = F_l.astype(np.float32)
F_k = F_k.astype(np.float32)
B_k = B_k.astype(np.float32)
B_v = B_v.astype(np.float32)

import IPython; IPython.embed()

BOLD = B_k.T
VASO = Y['vv'].T

fig = plt.figure(figsize=(7, 7))
plt.subplot(2, 1, 1)
plt.title(r'VASO signal (% from baseline)')
plt.imshow(VASO, aspect='auto', cmap='Blues', interpolation='none')
plt.colorbar()
plt.ylabel('Cortical depth (K)')
plt.subplot(2, 1, 2)
plt.title(r'BOLD signal (% from baseline)')
plt.imshow(BOLD, aspect='auto', cmap='Reds', interpolation='none')
plt.colorbar()
plt.ylabel('Cortical depth (K)')
plt.xlabel('Time')
plt.tight_layout(pad=1)
plt.savefig('png/BOLD_VASO.png', format='png', dpi=1200)
plt.show()

import IPython; IPython.embed()