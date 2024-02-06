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

PROB_K = I2K(K, species, area, sigma=K/10)

# flatten probabilities along last two dimensions
PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])

print('Selecting excitatory synapses...')
E_map = np.zeros((K, 80))
E_map[:, ::2] = 1
E_map[:, 1::2] = 1  # inhibitory contribution
E_map[:, 64:] = 1

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

# normalize between 0 and 0.6
# I_tot = (I_tot - np.min(I_tot)) / (np.max(I_tot) - np.min(I_tot)) * 0.8

F = NVC(I_tot)
F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps

B, _, _ = lbr.sim(F, K, integrator='numba')
# compute betas 

def HRF(t, peak_time=6, undershoot_time=10, peak_dispersion=1, undershoot_dispersion=1):
    peak_gaussian = np.exp(-0.5 * ((t - peak_time) / peak_dispersion) ** 2)
    undershoot_gaussian = np.exp(-0.5 * ((t - undershoot_time) / undershoot_dispersion) ** 2)
    hrf = peak_gaussian - undershoot_gaussian * 0.35
    return hrf

# Convolve I_th_T with HRF
I_th_T_convolved = np.zeros((T, M))
for i in range(M):
    I_th_T_convolved[:, i] = convolve(I_th_T[:, i], HRF(np.arange(0, 15, lbr_dt)))[:T]

X_th = I_th_T_convolved[::int(lbr_dt/dt), 2][300:]

X_th = np.tile(X_th, (K, 1)).T

beta = np.linalg.lstsq(X_th, B, rcond=None)[0]
BETA = beta[0]

# downsample to 5 depths
BETA_downsampled = np.zeros(5)  
for i in range(5):
    BETA_downsampled[i] = np.mean(BETA[i*6:(i+1)*6])

plt.figure()
plt.plot(BETA_downsampled)


# post-synaptic current per population
populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
colors = plt.cm.Spectral(np.linspace(0, 1, M))
tot_current = np.zeros(M)
for i in range(M):
    tot_current[i] = abs(np.sum(I[3000:, i*8:(i+1)*8]))

plt.figure(figsize=(5, 5))
plt.bar(populations, tot_current, color=colors)
plt.savefig('pdf/th_totcurrent.pdf')

# forward mapping of current to depth
plt.figure(figsize=(5, 5))
plt.plot(I_tot[int(7.5/dt)], np.arange(K))
plt.gca().invert_yaxis()
plt.savefig('pdf/th_I.pdf')
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(F[10000], np.arange(K))
plt.gca().invert_yaxis()
plt.savefig('pdf/th_F.pdf')

plt.figure(figsize=(5, 5))
plt.plot(BETA, np.arange(K))
plt.gca().invert_yaxis()
plt.savefig('pdf/th_BETA.pdf')


plt.figure()
colors = plt.cm.Spectral(np.linspace(0, 1, M))
populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
for i in range(M):
    plt.plot(B[:, i], color=colors[i], label=populations[i])
plt.legend()
plt.show()

import IPython; IPython.embed()
