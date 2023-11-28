import numpy as np
import h5py
import json

from models.DMF import DMF

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../')

import pylab as plt


"""
Generate dataset of neuronal response to external input in the microcircuit model.
"""

dt = 1e-4
t_sim = 1
T = int(t_sim / dt)
M = 8

J_E = 87.8e-3

I_th    = np.array([0,              0,                 0.0983*902*15*J_E, 0.0619*902*15*J_E, 0,              0,                0.0512*902*15*J_E, 0.0196*902*15*J_E])
I_cc    = np.array([0.1*1200*5*J_E, 0.085*1200*5*J_E,  0,                 0,                 0.1*1200*5*J_E, 0.085*1200*5*J_E, 0,                 0                ])

dt = 1e-4
I_th_T = np.zeros((T, M))
I_th_T[int(0.2/dt):int(0.5/dt), :] = I_th

I_cc_T = np.zeros((T, M))
I_cc_T[int(0.6/dt):int(0.9/dt), :] = I_cc

X, Y, I = DMF(I_th=I_th_T, I_cc=I_cc_T, area='V1')

species = 'macaque'
area    = 'V1'
K       = 31
# load mapping from I to K
with h5py.File('maps/I2K_{}_{}_K{}.h5'.format(species, area, K), 'r') as hf:
    PROB_K = hf['PROB'][:]

PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])
I_K = np.array([I[t] * PROB_K for t in range(T)])




plt.figure()
# effect of thalamic input
plt.subplot(2, 2, 1)
plt.imshow(np.sum(I_K[int(0.1/dt):, :, 64:72], axis=2).T, aspect='auto', cmap='Reds')
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Cortical Depth (K)')
plt.title('Thalamic input')

# effect of cortico-cortical input
plt.subplot(2, 2, 2)
plt.imshow(np.sum(I_K[int(0.1/dt):, :, 72:], axis=2).T, aspect='auto', cmap='Reds')
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Cortical Depth (K)')
plt.title('Cortico-cortical input')

# effect of recurrent input
plt.subplot(2, 2, 3)
plt.imshow(np.sum(I_K[int(0.1/dt):, :, ::2], axis=2).T, aspect='auto', cmap='Reds')
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Cortical Depth (K)')
plt.title('Recurrent input')

# effect of all input
plt.subplot(2, 2, 4)
plt.imshow(np.sum(I_K[int(0.1/dt):, :, ::2], axis=2).T, aspect='auto', cmap='Reds')
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Cortical Depth (K)')
plt.title('All input')

plt.tight_layout()

plt.show()

import IPython; IPython.embed()

