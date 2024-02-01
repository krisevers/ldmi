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

dt = 1e-4               # timestep in seconds
t_sim = 12              # simulation time in seconds
T = int(t_sim / dt)     # number of timesteps
M = 8                   # number of populations
L = 5                   # number of layers
layers = ['1', '23', '4', '5', '6']

species = 'macaque'     # species
area    = 'V1'          # area
K       = 31            # number of cortical depths

from maps.I2K import I2K, get_thickness
PROB_K = I2K(K, species, area, sigma=K/10)

# flatten probabilities along last two dimensions
PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])

E_map = np.zeros((K, 80))
E_map[:, ::2] = 1
E_map[:, 64:] = 1

CURRENTS    = np.zeros((M, L, K))
FLOWS       = np.zeros((M, L, K))
BETAS       = np.zeros((M, L, K))
for m in range(M):
    J_E = 87.8e-3

    dt = 1e-4
    I_th_T = np.zeros((T, M))
    I_th_T[int(2/dt):int(7/dt), m] = 0

    I_cc_T = np.zeros((T, M))
    I_cc_T[int(2/dt):int(7/dt), m] = 200

    _, _, I, F = DMF(I_th=I_th_T, I_cc=I_cc_T, area='V1', t_sim=t_sim, dt=dt)

    for l in range(L):
        print('Computing laminar projection of currents to synapses... (m=%d) (l=%d)' % (m, l), end='\r')
        PROB_KK = np.copy(PROB_K)
        th = get_thickness(K, species, area)
        for k in range(K):
            if th[k] != layers[l]:
                PROB_KK[k, 72:] = 0
            else:
                PROB_KK[k, 72:] = .05

        MAP = np.zeros((I.shape[0], K))
        for i in range(I.shape[0]):
            MAP[i] = (I[i] @ (PROB_KK * E_map).T)

        I_BASE = I[int(1/dt)] @ (PROB_KK * E_map).T

        from models.NVC import NVC
        from models.LBR import LBR


        tot_current = (MAP - I_BASE) / (np.max(MAP) + I_BASE)
        CURRENTS[m, l] = (tot_current).max(axis=0)

        dt = 1e-4
        lbr_dt = 0.001
        lbr = LBR(K)

        # F = NVC(MAP - I_BASE)
        F = NVC(tot_current[1000:])
        F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

        FLOWS[m, l] = F.max(axis=0)

        B, _, _ = lbr.sim(F, K, integrator='numba')

        # compute betas
        # beta = np.linalg.lstsq(X, B, rcond=None)[0]
        BETAS[m, l] = B.max(axis=0)


# postprocessing
# normalize between 0 and 1
# for m in range(M):
#     BETAS[m] = (BETAS[m] - np.min(BETAS[m])) / (np.max(BETAS[m]) - np.min(BETAS[m]))

vmax = np.max(BETAS)
vmin = np.min(BETAS)

plt.figure(figsize=(2, 10))
blues   = plt.cm.Blues(np.linspace(0, 1, L))
reds    = plt.cm.Reds(np.linspace(0, 1, L))
labels  = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
for m in range(M):
    plt.subplot(8, 1, m+1)
    for l in range(L):
        if m % 2 == 0:
            plt.plot(BETAS[m, l], np.arange(K), color=blues[l])
        else:
            plt.plot(BETAS[m, l], np.arange(K), color=reds[l])
    plt.gca().invert_yaxis()
            
    plt.ylabel('Cortical Depth (K)')
    plt.title(labels[m])
plt.tight_layout()
plt.savefig('pdf/cc_input.pdf', dpi=300)
plt.show()

# same plot but group depths together
K_grouped = 5
K_ratio   = int(K / K_grouped)
BETAS_grouped = np.zeros((M, L, K_ratio))
for m in range(M):
    for l in range(L):
        for k in range(K_ratio):
            BETAS_grouped[m, l, k] = BETAS[m, l, k*K_grouped:(k+1)*K_grouped].max()

plt.figure(figsize=(2, 10))
blues   = plt.cm.Blues(np.linspace(0, 1, L))
reds    = plt.cm.Reds(np.linspace(0, 1, L))
labels  = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
for m in range(M):
    plt.subplot(4, 2, m+1)
    for l in range(L):
        if m % 2 == 0:
            plt.plot(BETAS_grouped[m, l], np.arange(K_ratio), color=blues[l])
        else:
            plt.plot(BETAS_grouped[m, l], np.arange(K_ratio), color=reds[l])
    plt.gca().invert_yaxis()
            
    plt.ylabel('Cortical Depth (K)')
    plt.title(labels[m])
plt.tight_layout()
plt.savefig('pdf/cc_input_grouped.pdf', dpi=300)
plt.show()

import IPython; IPython.embed()

