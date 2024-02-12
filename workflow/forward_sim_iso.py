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

num_conditions = 8
num_levels     = 5
conditions = np.arange(M)                       # target populations
levels     = np.linspace(0, 100, num_levels)    # excitatory input levels

I_all = np.zeros((num_conditions, num_levels, K))
F_all = np.zeros((num_conditions, num_levels, K))
V_all = np.zeros((num_conditions, num_levels, K))
B_all = np.zeros((num_conditions, num_levels, K))

I_tot_all = np.zeros((num_conditions, num_levels, T-3000, K))
for c in range(num_conditions):
    for l in range(num_levels):
        print(f'Condition {c+1}/{num_conditions}, Level {l+1}/{num_levels}...', end='\r')

        J_E = 87.8e-3

        I_th_T = np.zeros((T, M))

        I_cc_T = np.zeros((T, M))
        I_cc_T[int(5/dt):, c] = levels[l]

        X, Y, I, F = DMF(I_th=I_th_T, I_cc=I_cc_T, t_sim=t_sim, area='V1')

        from maps.I2K import I2K

        PROB_K = I2K(K, species, area, sigma=K/15)

        # flatten probabilities along last two dimensions
        PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])


        E_map = np.zeros((K, 80))
        E_map[:, ::2]  = 1
        E_map[:, 1::2] = 0  # inhibitory contribution
        E_map[:, 64:]  = 1

        MAP = np.zeros((I.shape[0], K))
        for i in range(I.shape[0]):
            MAP[i] = (I[i] @ (PROB_K * E_map).T)

        CURRENT_BASE = I[3000] @ (PROB_K * E_map).T

        from models.NVC import NVC
        from models.LBR import LBR
        from scipy.signal import convolve

        dt = 1e-4
        lbr_dt = 0.001
        lbr = LBR(K)

        I_tot = MAP[3000:] - CURRENT_BASE

        I_tot *= 1e3

        I_tot_all[c, l] = I_tot

for c in range(num_conditions):
    for l in range(num_levels):
        # running mean
        I_tot = I_tot_all[c, l]
        I_tot = np.array([np.mean(I_tot[i:i+100], axis=0) for i in range(len(I_tot) - 100)])
        I_tot -= I_tot[0]
        I_tot = np.concatenate((I_tot, np.zeros((100, K))))     # add lost timesteps at end

        # normalize between 0 and 0.6
        I_tot = (I_tot - np.min(I_tot_all)) / (np.max(I_tot_all) - np.min(I_tot_all)) * 0.8

        F = NVC(I_tot)
        F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

        B, _, Y = lbr.sim(F, K, integrator='numba')

        # save max amplitudes at each condition
        I_all[c, l] = I_tot.max(axis=0)
        F_all[c, l] = F[-1]
        V_all[c, l] = Y['vv'][-1]
        B_all[c, l] = B[-1]


import IPython; IPython.embed()


print('Generating figure...')
fig, ax = plt.subplots(nrows=8, ncols=4, figsize=(13, 13))
greens  = plt.cm.Greens(np.linspace(0, 1, num_levels))
blues   = plt.cm.Blues(np.linspace(0, 1, num_levels))
purples = plt.cm.Purples(np.linspace(0, 1, num_levels))
reds    = plt.cm.Reds(np.linspace(0, 1, num_levels))

populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
for c in range(num_conditions):
    for l in range(num_levels):
        print(f'Condition {c+1}/{num_conditions}, Level {l+1}/{num_levels}...', end='\r')
        ax[c, 0].set_ylabel(populations[c], rotation=45)
        ax[c, 0].plot(np.flip(I_all[c, l]), np.arange(K), color=greens[l], lw=2)
        ax[c, 0].set_ylim(0, K-1)
        # ax[c, 0].set_xlim(np.min(I_all), np.max(I_all))
        ax[c, 0].set_yticks([K-1, 0], ['CSF', 'WM'])
        ax[c, 0].set_title('Transmembrane Currents')

        ax[c, 1].plot(np.flip(F_all[c, l]), np.arange(K), color=blues[l], lw=2)
        ax[c, 1].set_ylim(0, K-1)
        # ax[c, 1].set_xlim(np.min(F_all), np.max(F_all))
        ax[c, 1].set_yticks([K-1, 0], ['CSF', 'WM'])
        ax[c, 1].set_title('Blood Flow')

        ax[c, 2].plot(np.flip(V_all[c, l]), np.arange(K), color=purples[l], lw=2)
        ax[c, 2].set_ylim(0, K-1)
        # ax[c, 2].set_xlim(np.min(V_all), np.max(V_all))
        ax[c, 2].set_yticks([K-1, 0], ['CSF', 'WM'])
        ax[c, 2].set_title('Blood Volume')

        ax[c, 3].plot(np.flip(B_all[c, l]), np.arange(K), color=reds[l], lw=2)
        ax[c, 3].set_ylim(0, K-1)
        # ax[c, 3].set_xlim(np.min(B_all), np.max(B_all))
        ax[c, 3].set_yticks([K-1, 0], ['CSF', 'WM'])
        ax[c, 3].set_title('BOLD Signal (%)')

    cmaps = ['Greens', 'Blues', 'Purples', 'Reds']
    for i in range(4):
        for axis in ['top','bottom','left','right']:
            ax[c, i].spines[axis].set_linewidth(2)
            ax[c, i].tick_params(width=2)
            cax = ax[c, i].inset_axes([0.8, 0.8, 0.1, 0.05])
            cmap = plt.cm.ScalarMappable(cmap = plt.get_cmap(cmaps[i]))
            cbar = fig.colorbar(cmap, cax=cax, orientation='horizontal', aspect=0.1)
            cbar.outline.set_linewidth(2)
            cbar.ax.tick_params(width=2)

plt.tight_layout()
# plt.savefig('eps/iso.eps', format='eps', dpi=300)
plt.savefig('pdf/iso.pdf', format='pdf', dpi=300)
plt.savefig('png/iso.png', format='png', dpi=300)
plt.show()
