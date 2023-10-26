import numpy as np
import pylab as plt

from worker import F

J_E = 87.8e-3

theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,                                     # intrinsic neuronal parameters
            'I_L4E': 0.0983*902*18*J_E, 'I_L4I': 0.0619*902*18*J_E, 'I_L6E': 0.0512*902*20*J_E, 'I_L6I': 0.0196*902*20*J_E, # external input
            'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,                                                        # neurovascular coupling parameters
            'E_0v': 0.35, 'V_0t': 2, 'TE': 0.028}                                                                           # hemodynamic parameters

E = {'K': 22, 'area': 'V1', 'T': 30, 'onset': 10, 'offset': 20}   # experimental parameters

# default forward model
_, _, _, _, _, B_k, B_v = F(E, theta, test=True)  # forward model

# casting to float32
B_k = B_k.astype(np.float32)
B_v = B_v.astype(np.float32)

# adjust P
theta['P_L4EtoL23E'] = 0
_, _, _, _, _, B_k_P, B_v_P = F(E, theta, test=True)  # forward model

# casting to float32
B_k_P = B_k_P.astype(np.float32)
B_v_P = B_v_P.astype(np.float32)

# plot
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(B_k - B_k_P, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.title('B_k')
plt.subplot(122)
plt.imshow(B_v - B_k_P, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.title('B_v')
plt.show()