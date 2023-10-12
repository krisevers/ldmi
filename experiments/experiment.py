import numpy as np
import pylab as plt

from worker import worker

num_conditions = 5

theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'nu_E': 1, 'nu_I': 1, 'G_EE': 0.1009, 'G_EI': 0.1346, 'G_IE': -0.1689*4, 'G_II': -0.1371*4, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,
            'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,
            'tau_mtt': 2, 'tau_vs': 4, 'alpha': 0.32, 'E_0': 0.4, 'V_0': 4, 'eps': 0.0463, 'rho_0': .191, 'nu_0': 126.3, 'TE': 0.028}
    
    
lam_E = np.linspace(0, 1, num_conditions)
lam_I = np.linspace(0, 1, num_conditions)

BOLD_all = np.empty((num_conditions, num_conditions), dtype=object)

for i, val_E in enumerate(lam_E):
    for j, val_I in enumerate(lam_I):
        print('progress: %d/%d' % (i*num_conditions + j, num_conditions**2), end='\r')

        theta['lam_E'] = val_E
        theta['lam_I'] = val_I
        U, X, F, V, Q, BOLD = worker(theta)

        BOLD_all[i, j] = BOLD

colors = plt.cm.Reds(np.linspace(0, 1, num_conditions))
plt.figure(figsize=(10, 10))
for i in range(num_conditions):
    plt.plot(BOLD_all[i, 0], label='lam_E = %.2f' % lam_E[i], color=colors[i])
plt.legend()

plt.figure(figsize=(10, 10))
for i in range(num_conditions):
    plt.plot(BOLD_all[4, i], label='lam_I = %.2f' % lam_I[i], color=colors[i])
plt.legend()

plt.show()


import IPython; IPython.embed()