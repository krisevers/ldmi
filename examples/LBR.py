import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import axes3d

import os
import sys
sys.path.insert(0, os.getcwd())

from ldmi.hemodynamics.LBR import Model

import IPython

t_sim = 10.
dt = 1e-4

stim_start = 2
stim_end   = 3
stim_mag   = 1

K = 4
X = np.zeros((K, int(t_sim/dt)))
X[:, int(stim_start/dt):int(stim_end/dt)] = stim_mag


L = Model(X, dt=dt, times=np.arange(0, t_sim, dt))

L.CerebralBloodFlow()

cbf = L.get_CBF()

plt.figure()
plt.subplot(211)
plt.title('Neural Response')
plt.imshow(X, aspect='auto', cmap='gray')
plt.subplot(212)
plt.title('Cerebral Blood Flow')
plt.imshow(cbf, aspect='auto', cmap='gray')

plt.tight_layout()
plt.show()

IPython.embed()
