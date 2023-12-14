import numpy as np

from models.RDM import RDM


# parameters
T = 1
dt = 5e-4
dt_rec = 5e-4
Nrecord = 8
seed = 0

# parameters
tau_m   = 0.01     # membrane time constant (s)
tau_s_E = 0.0005   # synaptic time constant (s)
tau_s_I = 0.0005   # synaptic time constant (s)

# connectivity parameters
P = np.array( # connection probabilities
               [[0.1009, 0.1689, 0.0440, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
                [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
                [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
                [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
                [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
                [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
                [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
P[0, 2] *= 2

M = 8
N = np.array([20683,	5834,	21915,	5479,	4850,	1065,	14395,	2948])

C = np.log(1-P) / np.log(1 - 1/(N * N)) / N     # number of synapses

g = -4
J_E = 0.176
J_I = J_E * g

G = np.tile([J_E, J_I], (M, int(M/2))) * C    # synaptic conductances

I_bg = np.array([19.149, 20.362, 30.805, 28.069, 29.437, 29.33, 34.932, 32.081])

I_bg[4] *= 1.05


# external input
I_ext = {"onset":   np.array([0.0, 0.0, 0.3,  0.3,    0.0, 0.0, 0.3,   0.3  ]),
         "offset":  np.array([0.0, 0.0, 0.5,  0.5,    0.0, 0.0, 0.5,   0.5  ]),
         "I":       np.array([0.0, 0.0, 19.0, 11.964, 0.0, 0.0, 9.896, 3.788]) * 0}

params = {
    "M": M,
    "N": N,
    "mu": I_bg,
    "Delta_u":  np.ones(M) *  5.0,
    "c":        np.ones(M) * 10.0,
    "vreset":   np.ones(M) *  0.0,
    "vth":      np.ones(M) * 15.0,
    "tref":     np.ones(M) *  0.002,
    "delay":    np.ones(M) *  0.001,
    "tau_m":    np.ones(M) * tau_m,
    "tau_s":    np.tile([tau_s_E, tau_s_I], int(M/2)),
    "weights":  G,
}

# run simulation
Abar, A = RDM(T, dt, dt_rec, params, I_ext, Nrecord, seed)

# plot results
import pylab as plt

colors = plt.cm.Spectral(np.linspace(0, 1, M))
populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']

plt.figure(figsize=(8,4))
plt.subplot(121)
for i in range(M):
    plt.plot(Abar[i,100:], color=colors[i], label=populations[i])
    plt.text(Abar[:,100:].shape[1], Abar[i,-1], populations[i], color=colors[i])
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.subplot(122)    # mean firing rate
emp_rates = np.array([0.974, 2.861, 4.673, 5.65, 8.141, 9.013, 0.988, 7.53])
plt.bar(np.arange(M), emp_rates, color=colors, alpha=0.5, width=1.0, edgecolor='k')
plt.bar(np.arange(M), np.mean(A[:,100:], axis=1), color=colors, width=.8, edgecolor='k')
plt.xticks(np.arange(M), populations)
plt.xlabel('Population')
plt.ylabel('Mean firing rate (Hz)')
# plt.subplot(133)    # synchrony
# emp_sync = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# plt.bar(np.arange(M), emp_sync, color=colors, alpha=0.5, width=1.0, edgecolor='k')
# # Synchrony of multiunit spiking activity quantified by the variance of the spike count histogram (bin width 3 ms) divided by its mean.
# sync = np.zeros(M)
# for i in range(M):
#     hist, bins = np.histogram(A[i,100:], bins=np.arange(0, A.shape[1], int(0.003/dt_rec)))
#     sync[i] = np.var(hist) / np.mean(hist)
# plt.bar(np.arange(M), sync, color=colors, width=.8, edgecolor='k')
# plt.xticks(np.arange(M), populations)
# plt.xlabel('Population')
# plt.ylabel('Synchrony')
plt.tight_layout()
plt.show()

import IPython; IPython.embed()