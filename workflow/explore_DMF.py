import numpy as np
import pylab as plt

from scipy.signal import welch, csd

from models.DMF import DMF

t_sim = 1
dt = 1e-4

onset_th  = 0.25
offset_th = 0.75
onset_cc  = 0.25
offset_cc = 0.75

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
F_prestim  = np.mean(F[1000:int(onset_th/dt)-1000, :], axis=0)
F_poststim = np.mean(F[int(onset_th/dt)+1000:int(offset_th/dt), :], axis=0)

N = np.array([20683,	5834,	21915,	5479,	4850,	1065,	14395,	2948])

# # multiply by population size
F_prestim  *= N / np.sum(N)
F_poststim *= N / np.sum(N)

vmin = np.min([F_prestim, F_poststim])
vmax = np.max([F_prestim, F_poststim])

# create barplots
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.bar(np.arange(8), F_prestim,  color=colors)
plt.xticks(np.arange(8), populations)
plt.ylim([vmin, vmax])
plt.ylabel('firing rate (Hz)')
plt.title('pre-stimulus')
plt.subplot(122)
plt.bar(np.arange(8), F_poststim, color=colors)
plt.xticks(np.arange(8), populations)
plt.ylim([vmin, vmax])
plt.ylabel('firing rate (Hz)')
plt.title('post-stimulus')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.bar(1, np.sum(F_prestim[:2]),   color='k')
plt.bar(2, np.sum(F_prestim[2:4]),  color='k')
plt.bar(3, np.sum(F_prestim[4:]),   color='k')
plt.bar(4, np.sum(F_poststim[:2]),  color='r')
plt.bar(5, np.sum(F_poststim[2:4]), color='r')
plt.bar(6, np.sum(F_poststim[4:]),  color='r')
plt.xticks([1, 2, 3, 4, 5, 6], ['L23', 'L4', 'L56', 'L23', 'L4', 'L56'])
plt.ylabel('firing rate (Hz)')
plt.tight_layout()
plt.show()

# post-stimulus - pre-stimulus
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.bar(np.arange(8), F_poststim - F_prestim, color=colors)
plt.xticks(np.arange(8), populations)
plt.ylim([vmin, vmax])
plt.ylabel('firing rate (Hz)')
plt.title('post-stimulus - pre-stimulus')
plt.subplot(122)
plt.bar(1, np.sum(F_poststim[:2]) - np.sum(F_prestim[:2]),   color='k')
plt.bar(2, np.sum(F_poststim[2:4]) - np.sum(F_prestim[2:4]),  color='k')
plt.bar(3, np.sum(F_poststim[4:]) - np.sum(F_prestim[4:]),   color='k')
plt.xticks([1, 2, 3], ['L23', 'L4', 'L56'])
plt.ylabel('firing rate (Hz)')
plt.tight_layout()
plt.show()







# import IPython; IPython.embed()