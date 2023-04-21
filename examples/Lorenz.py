import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import axes3d

import os
import sys
sys.path.insert(0, os.getcwd())

from ldmi.models.Lorenz import Sim

# Lorenz paramters and initial conditions
sigma, beta, rho = 10, 2.667, 28
y0 = [0, 1, 1.05]

t_sim = 100.
dt = 1e-4

L = Sim(dt=dt, t_sim=t_sim, y=y0, rho=rho, sigma=sigma, beta=beta)

# Integrate the Lorenz equations
L.integrate('euler')
T = L.get_times()
X = L.get_states()

print("simulation done!")

X = np.asarray(X)

# Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
