import numpy as np
import pylab as plt

import os

import torch

import sbi.utils as utils

import IPython

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='data/', help='Set path to obtain data')
parser.add_argument('-n', '--name', help='Name of experiment')

args = parser.parse_args()

PATH = args.path + args.name + '/'

posterior = torch.load(PATH + 'posterior.pt')

# load test data
psi_test = np.load(PATH + 'psi_test.npy')
theta_test = np.load(PATH + 'theta_test.npy')

num_samples = 10000

perm_idx = 0

posterior.set_default_x(psi_test[perm_idx])
posterior_samples = posterior.sample((num_samples,))

from view import pairplot, marginal_correlation, marginal

keys = np.load(PATH + 'keys.npy', allow_pickle=True)

# check if directory exists
if not os.path.exists('pdf/' + args.name + '/'):
    os.makedirs('pdf/' + args.name + '/')

fig, ax = pairplot(posterior_samples, labels=keys)
plt.savefig('pdf/' + args.name + '/pairplot.pdf', dpi=300)

fig, ax = marginal_correlation(posterior_samples, labels=keys, figsize=(10, 10))
plt.savefig('pdf/' + args.name + '/marginal_correlation.pdf', dpi=300)

fig, ax = marginal(posterior_samples, labels=keys, figsize=(8, 12))
for i in range(len(keys)):
    ax[i].axvline(theta_test[perm_idx][i], color='r', linestyle='--')
plt.savefig('pdf/' + args.name + '/marginal.pdf', dpi=300)

IPython.embed()
