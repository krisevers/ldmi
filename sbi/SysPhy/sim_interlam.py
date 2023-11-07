import numpy as np
import pylab as plt

from worker import F

J_E = 87.8e-3

theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,                                     # intrinsic neuronal parameters
            'I_L4E': 0.0983*902*18*J_E, 'I_L4I': 0.0619*902*18*J_E, 'I_L6E': 0.0512*902*20*J_E, 'I_L6I': 0.0196*902*20*J_E, # external input
            'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,                                                        # neurovascular coupling parameters
            'E_0v': 0.35, 'V_0t': 2, 'TE': 0.028}                                                                           # hemodynamic parameters

E = {'K': 12, 'area': 'V1', 'T': 15, 'onset': 1, 'offset': 5}   # experimental parameters

# default forward model
print('condition: default', end='\r')
Psi_def, _, _, _, _, _, _ = F(E, theta, mode='full')  # forward model

num_conditions = 8
theta = np.tile(theta, (num_conditions, 1))
Psi   = np.empty(num_conditions, dtype=object)
keys  = ['P_L23EtoL23E', 'P_L4EtoL23E', 'P_L23EtoL5E', 'P_L23EtoL4I', 'P_L23EtoL6I', 'P_L5EtoL23I', 'P_L6EtoL4I', 'P_L5EtoL6E']

for i in range(num_conditions):
    print('condition: ' + keys[i] + ' (' + str(i+1) + '/' + str(num_conditions) + ')', end='\r')
    theta[i, 0][keys[i]] = 0
    Psi[i], _, _, _, _, _, _ = F(E, theta[i, 0], mode='full')  # forward model

# plot
plt.figure(figsize=(7, 7))
plt.plot(Psi_def['peak_amp_k'],   np.arange(E['K']),  color='k', label='default')
for i in range(num_conditions-1):
    plt.plot(Psi[i]['peak_amp_k'], np.arange(E['K']),  label=keys[i])
plt.xlabel('BOLD amplitude')
plt.ylabel('cortical depth (K)')
plt.legend()
plt.gca().invert_yaxis()
plt.show()

import IPython; IPython.embed()