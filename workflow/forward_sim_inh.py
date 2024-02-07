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
t_sim = 50
T = int(t_sim / dt)
M = 8

J_E = 87.8e-3

I_th    = np.array([0,              0,                 0.0983*902*15*J_E, 0.0619*902*15*J_E, 0,              0,                0.0512*902*15*J_E, 0.0196*902*15*J_E])
I_cc    = np.array([0.1*1200*5*J_E, 0.085*1200*5*J_E,  0,                 0,                 0.1*1200*5*J_E, 0.085*1200*5*J_E, 0,                 0                ])

I_th_T = np.zeros((T, M))
I_th_T[int(5/dt):int(30/dt), :] = I_th

I_cc_T = np.zeros((T, M))
# I_cc_T[int(0.6/dt):int(0.9/dt), :] = I_cc

X, Y, I, F = DMF(I_th=I_th_T, I_cc=I_cc_T, t_sim=t_sim, area='V1')

species = 'macaque'
area    = 'V1'
K       = 31

from maps.I2K import I2K

PROB_K = I2K(K, species, area, sigma=K/15)

# flatten probabilities along last two dimensions
PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])

num_conditions = 11
conditions = np.linspace(0, 1, num_conditions)

I_all = np.zeros((num_conditions, K))
F_all = np.zeros((num_conditions, K))
V_all = np.zeros((num_conditions, K))
B_all = np.zeros((num_conditions, K))

for c in range(num_conditions):
    print('Selecting excitatory synapses...')
    E_map = np.zeros((K, 80))
    E_map[:, ::2]  = 1
    E_map[:, 1::2] = conditions[c]  # inhibitory contribution
    E_map[:, 64:]  = 1

    print('Computing laminar projection of currents to synapses...')
    MAP = np.zeros((I.shape[0], K))
    for i in range(I.shape[0]):
        MAP[i] = (I[i] @ (PROB_K * E_map).T)

    print('Computing baseline currents...')
    CURRENT_BASE = I[3000] @ (PROB_K * E_map).T

    from models.NVC import NVC
    from models.LBR import LBR
    from scipy.signal import convolve

    dt = 1e-4
    lbr_dt = 0.001
    lbr = LBR(K)

    I_tot = MAP[3000:] - CURRENT_BASE

    I_tot *= 1e3

    # running mean
    I_tot = np.array([np.mean(I_tot[i:i+100], axis=0) for i in range(len(I_tot) - 100)])
    I_tot -= I_tot[0]
    I_tot = np.concatenate((I_tot, np.zeros((100, K))))     # add lost timesteps at end

    # normalize between 0 and 0.6
    I_tot = (I_tot - np.min(I_tot)) / (np.max(I_tot) - np.min(I_tot)) * 0.8

    F = NVC(I_tot)
    F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

    B, _, Y = lbr.sim(F, K, integrator='numba')

    # save max amplitudes at each condition
    I_all[c] = np.max(I_tot, axis=0)
    F_all[c] = np.max(F, axis=0)
    V_all[c] = np.max(Y['vv'], axis=0)
    B_all[c] = np.max(B, axis=0)


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13, 3))
greens  = plt.cm.Greens(np.linspace(0, 1, num_conditions))
blues   = plt.cm.Blues(np.linspace(0, 1, num_conditions))
purples = plt.cm.Purples(np.linspace(0, 1, num_conditions))
reds    = plt.cm.Reds(np.linspace(0, 1, num_conditions))
for c in range(num_conditions):
    ax[0].set_ylabel('Cortical Depth')
    ax[0].plot(np.flip(I_all[c]), np.arange(K), color=greens[c], lw=2)
    ax[0].set_ylim(0, K-1)
    ax[0].set_yticks([K-1, 0], ['CSF', 'WM'])
    ax[0].set_title('Transmembrane Currents')

    ax[1].plot(np.flip(F_all[c]), np.arange(K), color=blues[c], lw=2)
    ax[1].set_ylim(0, K-1)
    ax[1].set_yticks([K-1, 0], ['CSF', 'WM'])
    ax[1].set_title('Blood Flow')

    ax[2].plot(np.flip(V_all[c]), np.arange(K), color=purples[c], lw=2)
    ax[2].set_ylim(0, K-1)
    ax[2].set_yticks([K-1, 0], ['CSF', 'WM'])
    ax[2].set_title('Blood Volume')

    ax[3].plot(np.flip(B_all[c]), np.arange(K), color=reds[c], lw=2)
    ax[3].set_ylim(0, K-1)
    ax[3].set_yticks([K-1, 0], ['CSF', 'WM'])
    ax[3].set_title('BOLD Signal (%)')

cmaps = ['Greens', 'Blues', 'Purples', 'Reds']

for i in range(4):
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(2)
        ax[i].tick_params(width=2)
        cax = ax[i].inset_axes([0.05, 0.5, 0.1, 0.05])
        cmap = plt.cm.ScalarMappable(cmap = plt.get_cmap(cmaps[i]))
        cbar = fig.colorbar(cmap, cax=cax, orientation='horizontal', aspect=0.1)
        cbar.outline.set_linewidth(2)
        cbar.ax.tick_params(width=2)

# plt.tight_layout()
# plt.savefig('eps/th_inh.eps', format='eps', dpi=300)
plt.savefig('pdf/th_inh.pdf', format='pdf', dpi=300)
plt.savefig('png/th_inh.png', format='png', dpi=300)
plt.show()

import IPython; IPython.embed()
