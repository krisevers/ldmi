import numpy as np
import pylab as plt

from scipy.stats import gamma

"""
Test procedure for fitting gamma distributions to BOLD data
"""

# load BOLD data
BOLD = np.load('PDCM_BOLD.npy')
BOLD = BOLD[:,0]

# create stimulus
U = np.zeros((60000))
U[1000:5000] = 1

num_modes = 3

t = np.linspace(0, 30, 1000)

B = BOLD

alpha = np.zeros(num_modes)
loc = np.zeros(num_modes)
beta = np.zeros(num_modes)
x = np.zeros((num_modes, len(t)))
for i in range(num_modes):
    alpha[i], loc[i], beta[i] = gamma.fit(B)
    x[i] = gamma.pdf(t, alpha[i], loc=loc[i], scale=beta[i])
    # convolve with BOLD signal
    X = np.convolve(U, x[i], mode='same')

    B = B - X

plt.figure()
# plt.plot(BOLD, label='BOLD')
for i in range(num_modes):
    plt.plot(t, x[i], label='mode {}'.format(i))
plt.legend()
plt.show()

import IPython; IPython.embed()