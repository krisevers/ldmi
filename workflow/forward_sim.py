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

X, Y, I, F = DMF(I_th=I_th_T, I_cc=I_cc_T, area='V1')

species = 'macaque'
area    = 'V1'
K       = 31

from maps.I2K import I2K

PROB_K = I2K(K, species, area, sigma=1)

print('Selecting excitatory synapses...')
E_map = np.zeros((K, 80))
E_map[:, ::2] = 1
E_map[:, 64:] = 1

print('Computing laminar projection of currents to synapses...')
MAP = np.zeros((I.shape[0], K))
for i in range(I.shape[0]):
    MAP[i] = (I[i] @ (PROB_K * E_map).T)

print('Computing baseline currents...')
CURRENT_BASE = I[100] @ (PROB_K * E_map).T

from models.NVC import NVC
from models.LBR import LBR

dt = 1e-4
lbr_dt = 0.001
lbr = LBR(K)

F = NVC(I - CURRENT_BASE)
F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

B, _, _ = lbr.sim(F, K, integrator='numba')
# compute betas
beta = np.linalg.lstsq(X, B, rcond=None)[0]
BETA = beta[0]

plt.figure()

plt.show()

import IPython; IPython.embed()

