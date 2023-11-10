import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import axes3d

import os
import sys
sys.path.insert(0, os.getcwd())

from ldmi.models.DMF import Sim

# DMF parameters and initial conditions
sigma   = 0.02
tau_s   = 0.5e-3
tau_m   = 10e-3
C_m     = 250e-6 
tau_a   = 10. 
a       = 48.
b       = 981.
d       = 0.0089

M = 16
N = np.tile([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948], 2)

g = -4
J_E = 87.8e-3
J_I = J_E * g

J = np.tile([J_E, J_I, J_E, J_I, J_E, J_I, J_E, J_I], (M, 2))

P = np.array(
    [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
     [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
     [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
     [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
     [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
     [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
     [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
     [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
P[0, 2] *= 2
Z = np.zeros((8, 8))
P = np.hstack((np.vstack((P, Z)), np.vstack((Z, P))))


P[9,  0] = 0.1
P[1,  8] = 0.1
J[9,  0] = 0.18    # lateral inhibition
J[1,  8] = 0.18    # lateral inhibition
J[0,  0] = 0.13    # self-excitation
J[8,  8] = 0.13    # self-excitation

K = np.log(1-P) / np.log(1 - 1/(N * N.T)) / N

W = K * J

nu_bg = 8.
K_bg  = np.tile([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100], 2)
W_bg  = K_bg * J_E

nu_ext = np.zeros(M)
W_ext  = np.zeros(M, dtype=float)


layer = 'L6'
if layer == 'L23':
    idx = 0
elif layer == 'L4':
    idx = 2
elif layer == 'L5':
    idx = 4
elif layer == 'L6':
    idx = 6

nu = 0
K_ext  = 50
nu_ext[idx]     = nu
nu_ext[idx+8]   = nu
W_ext[idx]      = K_ext * J_E
W_ext[idx+8]    = K_ext * J_E

kappa = np.tile([1.5, 0., 0., 0., 0., 0., 0., 0.], 2)

y0 = np.zeros((M, 4), dtype=float)

t_sim = 30
dt = 1e-4

L = Sim(dt=dt, 
        t_sim=t_sim, 
        y=list(y0), 
        sigma=sigma, 
        tau_s=tau_s, 
        tau_m=tau_m, 
        C_m=C_m, 
        kappa=list(kappa), 
        tau_a=tau_a, 
        a=a, 
        b=b, 
        d=d, 
        nu_bg=nu_bg, 
        W_bg=list(W_bg), 
        nu_ext=list(nu_ext), 
        W_ext=list(W_ext), 
        W=list(W), 
        progress=False)

# Integrate the Lorenz equations
L.integrate('rk4')
T = L.get_times()
X = L.get_states()

X = np.asarray(X)

# Plot the population rates
population = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
colors = plt.cm.Spectral(np.linspace(0, 1, M))
fig = plt.figure()
for i in range(8):
    plt.subplot(4, 2, i+1)
    plt.title(population[i])
    plt.plot(T[500:], X[500:, i, 3],     color='red')
    plt.plot(T[500:], X[500:, i+8, 3],   color='blue')
    plt.xlabel('Time')
    plt.ylabel('Rate [Hz]')
plt.tight_layout()
plt.show()