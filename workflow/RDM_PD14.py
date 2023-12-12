import numpy as np

from models.RDM import RDM


# parameters
T = 1
dt = 5e-4
dt_rec = 1e-3
Nrecord = 8
seed = 0

# parameters
tau_m   = 0.02    # membrane time constant (s)
tau_s_E = 0.003   # synaptic time constant (s)
tau_s_I = 0.006   # synaptic time constant (s)

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

g = -4.
J_E = 8.78e-3
J_I = J_E * g

G = np.tile([J_E, J_I], (M, int(M/2))) * C    # synaptic conductances

C_bg = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])   # number of background synapses
G_bg = C_bg * J_E
nu_bg = 8.
I_bg = G_bg * nu_bg

I_bg = np.array([19.149, 20.362, 30.805, 28.069, 29.437, 29.33, 34.932, 32.081])

# external input
I_ext = {"onset":   np.array([0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]),
         "offset":  np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
         "I":       np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}

params = {
    "M": M,
    "N": N,
    "mu": I_bg,
    "Delta_u": np.ones(M) * 5.0,
    "c": np.ones(M) * 15.0,
    "vreset": np.zeros(M),
    "vth": np.ones(M) * 15.0,
    "tref": np.ones(M) * 0.002,
    "delay": np.zeros(M),
    "tau_m": np.ones(M) * tau_m,
    "tau_s": np.tile([tau_s_E, tau_s_I], int(M/2)),
    "weights": G,
}

# run simulation
Abar, A = RDM(T, dt, dt_rec, params, I_ext, Nrecord, seed)

# plot results
import pylab as plt

colors = plt.cm.Spectral(np.linspace(0, 1, M))
populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']

plt.figure()
plt.subplot(211)
for i in range(M):
    plt.plot(Abar[i,:], color=colors[i], label=populations[i])
    plt.text(Abar.shape[1], Abar[i,-1], populations[i])
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.subplot(212)
plt.bar(np.arange(M), np.mean(Abar[:,100:], axis=1), color=colors)
plt.xticks(np.arange(M), populations)
plt.xlabel('Population')
plt.ylabel('Mean firing rate (Hz)')
plt.show()
