import numpy as np
import pylab as plt

import torch

import sbi.utils as utils

import IPython

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='data', help='Set path to obtain data')

args = parser.parse_args()

PATH = args.path

posterior = torch.load(PATH + '/PDCM_posterior.pt')


obs_theta = np.array([0.6, 1.5, 0.6, 2, 4, 0.32, 0.4, 2, 0.2, 0.191, 126.3, 0.028])
obs_x = np.array([1.5555508322922773, 1.479733471762112, -0.0732075162889782, 1.127864969275166, 3.189754360979834, -0.2541013394438985, 10.806000000000001, 42.606, 0.0007804106206767969, -0.0005311255055309161, 3.747, 34.845, 95.39888544630811, -2.0658355087714564, -46.17932310740531])
num_samples = 10000

posterior.set_default_x(obs_x)
posterior_samples = posterior.sample((num_samples,))



from view import pairplot, marginal_correlation, marginal

keys = np.array([r'$c_{1}$', r'$c_{2}$', r'$c_{3}$', r'$\tau_{mtt}$', r'$\tau_{vs}$', 
                    r'$\alpha$', r'$E_0$', r'$V_0$', r'$\epsilon$', r'$\rho_0$', 
                    r'$\nu_0$', r'$TE$'])

limits = np.array([
    [0.3,       0.9   ],
    [1,         2     ],
    [0.3,       0.9   ],
    [1,         5     ],
    [0.1,      30     ],
    [0.1,       0.5   ],
    [0.1,       0.8   ],
    [1,        10     ],
    [0.3390,    0.3967],
    [10,     1000     ],
    [40,      440     ],
    [0.015,     0.040 ],
])


fig, ax = pairplot(posterior_samples, labels=keys)
plt.savefig('svg/pairplot.svg', dpi=300)

fig, ax = marginal_correlation(posterior_samples, labels=keys, figsize=(10, 10))
plt.savefig('svg/marginal_correlation.svg', dpi=300)

# NVC
fig, ax = pairplot(posterior_samples[:, :3], labels=keys[:3])
plt.savefig('svg/pairplot_NVC.svg', dpi=300)
# BOLD 
fig, ax = pairplot(posterior_samples[:, 3:], labels=keys[3:])
plt.savefig('svg/pairplot_BOLD.svg', dpi=300)

fig, ax = marginal(posterior_samples, labels=keys, limits=limits, figsize=(8, 12))
for i in range(len(keys)):
    ax[i].axvline(obs_theta[i], color='r', linestyle='--')
plt.savefig('svg/marginal.svg', dpi=300)

IPython.embed()
