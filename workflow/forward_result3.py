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
P = 4                   # number of layers receiving input
INH = 21                # number of inhibitory contributions
inh_contr = np.linspace(0, 1, INH)

species = 'macaque'     # species
area    = 'V1'          # area
K       = 31            # number of cortical depths

from maps.I2K import I2K, get_thickness
PROB_K = I2K(K, species, area, sigma=K/10)

# flatten probabilities along last two dimensions
PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])

E_map = np.zeros((K, 80))
E_map[:, ::2]  = 1
E_map[:, 64:]  = 1

CURRENTS    = np.zeros((P, INH, K))
FLOWS       = np.zeros((P, INH, K))
BETAS       = np.zeros((P, INH, K))
for idp, p in enumerate(np.arange(0, M, 2)):
    J_E = 87.8e-3

    dt = 1e-4
    I_th_T = np.zeros((T, M))
    I_th_T[int(2.5/dt):int(7/dt), p]   = 0
    I_th_T[int(2.5/dt):int(7/dt), p+1] = 0

    I_cc_T = np.zeros((T, M))
    I_cc_T[int(2.5/dt):int(7/dt), p]   = 1000
    I_cc_T[int(2.5/dt):int(7/dt), p+1] = 800

    _, _, I, F = DMF(I_th=I_th_T, I_cc=I_cc_T, area='V1', t_sim=t_sim, dt=dt)

    for inh in range(INH):
        # print('Computing laminar projection of currents to synapses... (p=%d) (l=%d)' % (p, l), end='\r')
        PROB_KK = np.copy(PROB_K)
        th = get_thickness(K, species, area)

        PROB_KK[:, 72:] = 0

        E_map[:, 1::2] = inh_contr[inh]

        MAP = np.zeros((I.shape[0], K))
        for i in range(I.shape[0]):
            MAP[i] = (I[i] @ (PROB_KK * E_map).T)

        I_BASE = I[int(2/dt)] @ (PROB_KK * E_map).T

        from models.NVC import NVC
        from models.LBR import LBR

        MAP = MAP[int(2/dt):]

        tot_current = (MAP - I_BASE) / (np.max(MAP, axis=0))
        print(tot_current.sum())
        CURRENTS[idp, inh] = (tot_current).max(axis=0)

        dt = 1e-4
        lbr_dt = 0.001
        lbr = LBR(K)

        # F = NVC(MAP - I_BASE)
        F = NVC(tot_current)
        F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

        FLOWS[idp, inh] = F.max(axis=0)

        B, _, _ = lbr.sim(F, K, integrator='numba')

        # compute betas
        # beta = np.linalg.lstsq(X, B, rcond=None)[0]
        BETAS[idp, inh] = B.max(axis=0)


# postprocessing
# normalize between 0 and 1
# for m in range(M):
#     BETAS[m] = (BETAS[m] - np.min(BETAS[m])) / (np.max(BETAS[m]) - np.min(BETAS[m]))

vmax = np.max(BETAS)
vmin = np.min(BETAS)

plt.figure(figsize=(2, 10))
blues   = plt.cm.Blues(np.linspace(0, 1, INH))
reds    = plt.cm.Reds(np.linspace(0, 1, INH))
labels  = ['L23', 'L4', 'L5', 'L6']
for p in range(P):
    plt.subplot(4, 1, p+1)
    for inh in range(INH):
            plt.plot(BETAS[p, inh], np.arange(K), color=blues[inh])
    plt.gca().invert_yaxis()
            
    plt.ylabel('Cortical Depth (K)')
    plt.title(labels[p])
plt.tight_layout()
plt.savefig('pdf/cc_input_player.pdf', dpi=300)
plt.show()

# # same plot but group depths together
# K_grouped = 5
# K_ratio   = int(K / K_grouped)
# BETAS_grouped = np.zeros((M, L, K_ratio))
# for p in range(P):
#     for l in range(L):
#         for k in range(K_ratio):
#             BETAS_grouped[p, l, k] = BETAS[p, l, k*K_grouped:(k+1)*K_grouped].max()

# plt.figure(figsize=(2, 10))
# blues   = plt.cm.Blues(np.linspace(0, 1, L))
# reds    = plt.cm.Reds(np.linspace(0, 1, L))
# labels  = ['L23', 'L4', 'L5', 'L6']
# for p in range(P):
#     plt.subplot(4, 1, p+1)
#     for l in range(L):
#         plt.plot(BETAS_grouped[p, l], np.arange(K_ratio), color=blues[l])
#     plt.gca().invert_yaxis()
            
#     plt.ylabel('Cortical Depth (K)')
#     plt.title(labels[p])
# plt.tight_layout()
# plt.savefig('pdf/cc_input_player_grouped.pdf', dpi=300)
# plt.show()

# plt.figure(figsize=(5, 5))
# spectral   = plt.cm.Spectral(np.linspace(0, 1, P))
# labels = ['L23', 'L4', 'L5', 'L6']
# for p in range(P):
#     plt.plot((BETAS_grouped[p, 0]-BETAS_grouped[p, 0].mean())/BETAS_grouped[p, 0].max(), np.arange(K_ratio), color=spectral[p], label=labels[p])
# plt.gca().invert_yaxis()
# plt.ylabel('Cortical Depth (K)')
# plt.title('Layer Profile to Layer Specific External Input')
# plt.legend()
# plt.tight_layout()
# plt.savefig('pdf/cc_input_player_grouped_all.pdf', dpi=300)
# plt.show()

import IPython; IPython.embed()

