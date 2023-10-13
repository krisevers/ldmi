import numpy as np
import pylab as plt

import torch

import sbi.utils as utils

import IPython

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='data/', help='Set path to obtain data')
parser.add_argument('-n', '--name', help='Name of experiment')

args = parser.parse_args()

PATH = args.path

posterior = torch.load(PATH + args.name + '_posterior.pt')

# load test data
psi_test = np.load(PATH + args.name + '_psi_test.npy')
theta_test = np.load(PATH + args.name + '_theta_test.npy')

num_samples = 10000

perm_idx = 0

posterior.set_default_x(psi_test[perm_idx])
posterior_samples = posterior.sample((num_samples,))



from view import pairplot, marginal_correlation, marginal

keys = ['I_L23E', 'I_L23I', 'I_L4E', 'I_L4I', 'I_L5E', 'I_L5I', 'I_L6E', 'I_L6I']

fig, ax = pairplot(posterior_samples, labels=keys)
plt.savefig('pdf/pairplot.pdf', dpi=300)

fig, ax = marginal_correlation(posterior_samples, labels=keys, figsize=(10, 10))
plt.savefig('pdf/marginal_correlation.pdf', dpi=300)

# NVC
fig, ax = pairplot(posterior_samples[:, :3], labels=keys[:3])
plt.savefig('pdf/pairplot_NVC.pdf', dpi=300)
# BOLD 
fig, ax = pairplot(posterior_samples[:, 3:], labels=keys[3:])
plt.savefig('pdf/pairplot_BOLD.pdf', dpi=300)

fig, ax = marginal(posterior_samples, labels=keys, figsize=(8, 12))
for i in range(len(keys)):
    ax[i].axvline(theta_test[perm_idx], color='r', linestyle='--')
plt.savefig('pdf/marginal.pdf', dpi=300)

IPython.embed()
