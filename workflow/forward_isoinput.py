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
t_sim = 30
T = int(t_sim / dt)
M = 8


species = 'macaque'
area    = 'V1'
K       = 31

from maps.I2K import I2K
PROB_K = I2K(K, species, area, sigma=K/10)

# flatten probabilities along last two dimensions
PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])

E_map = np.zeros((K, 80))
E_map[:, ::2] = 1
E_map[:, 64:] = 1

BETAS = np.zeros((M, K))
for m in range(M):
    print('Computing laminar projection of currents to synapses... (m=%d)' % m, end='\r')
    J_E = 87.8e-3

    dt = 1e-4
    I_th_T = np.zeros((T, M))
    I_th_T[int(5/dt):int(10/dt), m] = 200

    I_cc_T = np.zeros((T, M))

    _, _, I, F = DMF(I_th=I_th_T, I_cc=I_cc_T, area='V1', t_sim=t_sim, dt=dt)

    MAP = np.zeros((I.shape[0], K))
    for i in range(I.shape[0]):
        MAP[i] = (I[i] @ (PROB_K * E_map).T)

    I_BASE = I[int(1/dt)] @ (PROB_K * E_map).T

    from models.NVC import NVC
    from models.LBR import LBR

    dt = 1e-4
    lbr_dt = 0.001
    lbr = LBR(K)

    F = NVC(MAP - I_BASE)
    F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

    B, _, _ = lbr.sim(F, K, integrator='numba')

    # compute betas
    # beta = np.linalg.lstsq(X, B, rcond=None)[0]
    BETAS[m] = B.max(axis=0)


# postprocessing
# normalize between 0 and 1
# for m in range(M):
#     BETAS[m] = (BETAS[m] - np.min(BETAS[m])) / (np.max(BETAS[m]) - np.min(BETAS[m]))

vmax = np.max(BETAS)
vmin = np.min(BETAS)

plt.figure()
plt.suptitle('K = %d' % K)
colors = plt.cm.Spectral(np.linspace(0, 1, M))
labels = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
for m in range(M):
    plt.subplot(2, 4, m+1)
    plt.barh(y=np.arange(K), width=BETAS[m], height=1, color=colors[m])
    plt.gca().invert_yaxis()
    plt.ylabel('Cortical depth (K)')
    plt.title(labels[m])
plt.tight_layout()
plt.savefig('pdf/isoinput.pdf', dpi=300)
plt.show()

# same plot but group depths together
K_grouped = 5
K_ratio   = int(K / K_grouped)
BETAS_grouped = np.zeros((M, K_ratio))
for m in range(M):
    for k in range(K_ratio):
        BETAS_grouped[m, k] = BETAS[m, k*K_grouped:(k+1)*K_grouped].max()

plt.figure()
plt.suptitle('K = %d' % K_grouped)
colors = plt.cm.Spectral(np.linspace(0, 1, M))
labels = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
for m in range(M):
    plt.subplot(2, 4, m+1)
    plt.barh(y=np.arange(K_ratio), width=BETAS_grouped[m], height=1, color=colors[m])
    plt.gca().invert_yaxis()
    plt.ylabel('Cortical depth (K)')
    plt.title(labels[m])
plt.tight_layout()
plt.savefig('pdf/isoinput_grouped.pdf', dpi=300)
plt.show()



import IPython; IPython.embed()

