import numpy as np
import pylab as plt

from scipy.signal import welch, csd

from models.DMF import DMF

t_sim = 1
dt = 1e-4

onset_th  = 0.25
offset_th = 0.30
onset_cc  = 0.25
offset_cc = 0.30

J_E = 87.8e-3   # synaptic strength
P_th    = np.array([0,   0,     0.0983,   0.0619, 0,   0,     0.0512, 0.0196])  # thalamic connection probabilities
P_cc    = np.array([0.1, 0.085, 0,        0,      0.1, 0.085, 0.0,    0.0   ])  # cortico-cortical connection probabilities


N_th    = 902   # number of thalamic neurons
nu_th   = 15    # thalamic firing rate

N_cc    = 1200   # number of cortico-cortical neurons
nu_cc   = 0      # cortico-cortical firing rate

I_th = np.zeros((int(t_sim / dt), 8))   # thalamic input current
I_cc = np.zeros((int(t_sim / dt), 8))   # cortico-cortical input current

I_th[int(onset_th / dt):int(offset_th / dt)]   = P_th * N_th * nu_th * J_E

I_cc[int(onset_cc / dt):int(offset_cc / dt)]   = P_cc * N_cc * nu_cc * J_E

X, Y, I, F = DMF(I_th, I_cc, area='V1', t_sim=1, dt=1e-4, sigma=0.0)

colors = plt.cm.Spectral(np.linspace(0, 1, 8))
populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']




# group L23E and L23I
# group L4E and L4I
# group L5E, L5I, L6E and L6I
# F_prestim  = np.mean(F[1000:int(onset_th/dt)-1000, :], axis=0)
# F_poststim = np.mean(F[int(onset_th/dt)+1000:int(offset_th/dt), :], axis=0)

# N = np.array([20683,	5834,	21915,	5479,	4850,	1065,	14395,	2948])

# remove transients
F = F[1000:, :]

# using procedure from Einevoll et al. (2007) to generate spatio-temporal mapping of the population firing rates to the LFP
num_pop = 8
num_channels = 18
distance_between_channels = 100e-6 # m
num_time = np.shape(F)[0]

tau = 16.4e-3
delay = 0.6e-3
times = np.arange(num_time) * dt
h = [1/tau * np.exp(-(t - delay)/tau) for t in times]

# spatial profile
L = np.zeros((num_channels, num_pop))

channel_pos = np.arange(num_channels) * distance_between_channels
def spatial_profile(size, z, a, b=.1e-3):
    np.zeros(size)

    # symmetrix trapzoidal profile
    z_min    = z-a/2        # left up corner
    z_plus   = z+a/2        # right up corner
    z_min_0  = z_min - b    # left down corner
    z_plus_0 = z_plus - b   # right down corner

    dx = 1e-6

import IPython; IPython.embed()