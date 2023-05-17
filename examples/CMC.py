import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import axes3d

import os
import sys
sys.path.insert(0, os.getcwd())

"""
Implementation of the Canonical Microcircuit (CMC) model from Van De Steen et al. (2022).

x1_dot = x2
x2_dot = 1/tau1 * (U - G*f(x) - 2x2 - 1/tau1 * x1)

U: external input
G: connectivity matrix
f: sigmoidal activation function
tau: time constant
x1: membrane potential
x2: adaptation variable

f(x) = 1 / (1 + exp(-rx)) - 0.5
r: steepness of the sigmoidal activation function

"""

from ldmi.models.CMC import Sim

# CMC parameters and initial conditions
M = 4

G = [[-1, -2, -4,  0],  # intrinsic connectivity matrix
     [1,  -2, -4,  0],
     [1,   0, -4,  8],
     [0,   0, -4, -8]]
U = [1, 0.5, 1, 0.5]    # external input
tau = [2, 2, 16, 28]    # time constant


y0 = np.zeros((M, 2), dtype=float)

t_sim = 1
dt = 1e-4

L = Sim(dt=dt, t_sim=t_sim, y=list(y0), G=G, U=U, tau=tau)

# Integrate the Lorenz equations
L.integrate('euler')
T = L.get_times()
X = L.get_states()

X = np.asarray(X)

# Plot the population rates
population = ['SS', 'SP', 'II', 'DP']
colors = plt.cm.Spectral(np.linspace(0, 1, M))
fig = plt.figure()
for i in range(M):
    plt.plot(T, X[:, i, 3], label=population[i], color=colors[i])
plt.legend()
plt.xlabel('Time')
plt.ylabel('Rate [Hz]')

plt.figure()
colors = np.tile(['b', 'r'], 4)
plt.title("Mean Rates [Hz]")
plt.barh(np.arange(M), np.mean(X[:, :, 3], axis=0), color=colors)
plt.yticks(np.arange(M), population)
plt.xticks(np.arange(0, 10, 3))
plt.xlabel('Rate [Hz]')
plt.ylabel('Population')
plt.gca().invert_yaxis()
plt.show()
