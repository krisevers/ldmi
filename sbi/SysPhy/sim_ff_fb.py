import numpy as np
import pylab as plt

from worker import F

J_E = 87.8e-3

P_ff = [0,   0,     0.0983, 0.0619, 0,   0,     0.0512, 0.0196]
P_fb = [0.1, 0.085, 0,      0,      0.1, 0.085, 0.1,    0.085 ]

N_ff = 902
N_fb = 1200

nu_ff = 15 # 15
nu_fb = 5  # 5

L23, L4, L5, L6 = 0, 1, 1, 1

theta = {
        'I_L23E': P_fb[0]*N_fb*nu_fb*J_E*L23 + P_ff[0]*N_ff*nu_ff*J_E*L23,
        'I_L23I': P_fb[1]*N_fb*nu_fb*J_E*L23 + P_ff[1]*N_ff*nu_ff*J_E*L23,
        'I_L4E':  P_fb[2]*N_fb*nu_fb*J_E*L4  + P_ff[2]*N_ff*nu_ff*J_E*L4,
        'I_L4I':  P_fb[3]*N_fb*nu_fb*J_E*L4  + P_ff[3]*N_ff*nu_ff*J_E*L4,
        'I_L5E':  P_fb[4]*N_fb*nu_fb*J_E*L5  + P_ff[4]*N_ff*nu_ff*J_E*L5,
        'I_L5I':  P_fb[5]*N_fb*nu_fb*J_E*L5  + P_ff[5]*N_ff*nu_ff*J_E*L5,
        'I_L6E':  P_fb[6]*N_fb*nu_fb*J_E*L6  + P_ff[6]*N_ff*nu_ff*J_E*L6,
        'I_L6I':  P_fb[7]*N_fb*nu_fb*J_E*L6  + P_ff[7]*N_ff*nu_ff*J_E*L6,

        'lam_E': 10,
        'lam_I': 0
           }

E = {'K': 12, 'area': 'V1', 'T': 50, 'onset': 10, 'offset': 20}   # experimental parameters

Psi, X, S, F_l, F_k, B_k, B_v = F(E, theta, test=True)  # forward model

# cast to float32
F_l = F_l.astype(np.float32)
F_k = F_k.astype(np.float32)
B_k = B_k.astype(np.float32)
B_v = B_v.astype(np.float32)

plt.figure(figsize=(5, 5))
plt.subplot(211)
plt.imshow(B_k.T, aspect='auto', cmap='Reds', interpolation='none')
plt.title('BOLD Signal before downsampling (K x T)')
plt.colorbar()
plt.ylabel('Cortical Depth (K)')
plt.yticks(np.arange(1, E['K'], 3), np.arange(1, E['K'], 3))
plt.xticks([])
plt.subplot(212)
plt.imshow(B_v.T, aspect='auto', cmap='Reds', interpolation='none')
plt.title('BOLD Signal after downsampling (V x T)')
plt.colorbar()
plt.xlabel('Time (s)')
plt.yticks(np.arange(3), ['Superficial', 'Granular', 'Deep'])
plt.xticks(np.linspace(0, int((E['T']-1)/0.001), 3), np.linspace(0, E['T'], 3))
plt.tight_layout()
plt.savefig('png/BOLD_ff_deep.png', dpi=1200)
plt.show()

import IPython; IPython.embed()