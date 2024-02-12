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
Forward simulation of the microcircuit model with inhibitory input contribution to the laminar BOLD response.
"""

dt = 1e-4
t_sim = 30
T = int(t_sim / dt)
M = 8

species = 'macaque'
area    = 'V1'
K       = 31


file_path = 'data/dmf/all/data.h5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as file:
    # Access the datasets and attributes in the file
    THETA       = file['THETA'][()]
    PSI         = file['PSI'][()]
    BASELINE    = file['BASELINE'][()]

h5py.File(file_path, 'r').close()

from maps.I2K import I2K

PROB_K = I2K(K, species, area, sigma=K/15)

# flatten probabilities along last two dimensions
PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])


print('Selecting excitatory synapses...')
E_map = np.zeros((K, 80))
E_map[:, ::2]  = 1
E_map[:, 1::2] = 0  # inhibitory contribution
E_map[:, 64:]  = 1

I_base = np.load('data/dmf/I_base.npy')

PSI -= I_base

print('Computing laminar projection of currents to synapses...')
MAP = np.zeros((PSI.shape[0], K))
for i in range(PSI.shape[0]):
    MAP[i] = (PSI[i] @ (PROB_K * E_map).T)



import IPython; IPython.embed()





# for c in range(num_conditions):
#     for l in range(num_levels):
#         print(f'Condition {c+1}/{num_conditions}, Level {l+1}/{num_levels}...', end='\r')

#         J_E = 87.8e-3

#         I_th_T = np.zeros((T, M))

#         I_cc_T = np.zeros((T, M))
#         I_cc_T[int(5/dt):, c] = levels[l]

#         X, Y, I, F = DMF(I_th=I_th_T, I_cc=I_cc_T, t_sim=t_sim, area='V1')

#         from maps.I2K import I2K

#         PROB_K = I2K(K, species, area, sigma=K/15)

#         # flatten probabilities along last two dimensions
#         PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])


#         print('Selecting excitatory synapses...')
#         E_map = np.zeros((K, 80))
#         E_map[:, ::2]  = 1
#         E_map[:, 1::2] = 0  # inhibitory contribution
#         E_map[:, 64:]  = 1

#         print('Computing laminar projection of currents to synapses...')
#         MAP = np.zeros((I.shape[0], K))
#         for i in range(I.shape[0]):
#             MAP[i] = (I[i] @ (PROB_K * E_map).T)

#         print('Computing baseline currents...')
#         CURRENT_BASE = I[3000] @ (PROB_K * E_map).T

#         from models.NVC import NVC
#         from models.LBR import LBR
#         from scipy.signal import convolve

#         dt = 1e-4
#         lbr_dt = 0.001
#         lbr = LBR(K)

#         I_tot = MAP[3000:] - CURRENT_BASE

#         I_tot *= 1e3

#         # running mean
#         I_tot = np.array([np.mean(I_tot[i:i+100], axis=0) for i in range(len(I_tot) - 100)])
#         I_tot -= I_tot[0]
#         I_tot = np.concatenate((I_tot, np.zeros((100, K))))     # add lost timesteps at end

#         # normalize between 0 and 0.6
#         I_tot = (I_tot - np.min(I_tot)) / (np.max(I_tot) - np.min(I_tot)) * 0.8

#         F = NVC(I_tot)
#         F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

#         B, _, Y = lbr.sim(F, K, integrator='numba')

#         # save max amplitudes at each condition
#         I_all[c, l] = I_tot[-1]
#         F_all[c, l] = F[-1]
#         V_all[c, l] = Y['vv'][-1]
#         B_all[c, l] = B[-1]


import IPython; IPython.embed()

