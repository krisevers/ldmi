import numpy as np
import h5py
import pylab as plt

import os

import torch

import IPython

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',              default='data',    help='Set path to obtain data')
parser.add_argument('-n', '--name',                                 help='Name of experiment')
parser.add_argument('--num_samples',   type=int, default=1000,      help='Number of samples to draw from posterior')
parser.add_argument('--perm_idx',      type=int, default=0,         help='Index of test sample to plot')
parser.add_argument('-q', '--quality',     action='store_true',     help='Run quality control')
parser.add_argument('-e', '--empirical',   action='store_true',     help='Run empirical data')
args = parser.parse_args()

PATH = args.path + '/' + args.name + '/'

posterior = torch.load(PATH + 'posterior.pt')

# load train and test data
hf = h5py.File(PATH + 'data.h5', 'r')
THETA_train = np.array(hf.get('THETA_train'))
BETA_train   = np.array(hf.get('BETA_train'))
THETA_test  = np.array(hf.get('THETA_test'))
BETA_test    = np.array(hf.get('BETA_test'))
hf.close()

K = np.shape(THETA_train)[1]

num_samples = args.num_samples

perm_idx = args.perm_idx
posterior.set_default_x(BETA_test[perm_idx])
posterior_samples = posterior.sample((num_samples,))


from view import pairplot, marginal_correlation, marginal

keys = np.arange(K)

# check if directory exists
if not os.path.exists(PATH + 'pdf/'):
    os.makedirs(PATH + 'pdf/')

fig, ax = pairplot(posterior_samples.numpy(), labels=keys, figsize=(20, 20))
plt.savefig(PATH + 'pdf/pairplot.pdf', dpi=300)

fig, ax = marginal_correlation(posterior_samples.numpy(), labels=keys, figsize=(10, 10))
plt.savefig(PATH + 'pdf/marginal_correlation.pdf', dpi=300)

fig, ax = marginal(posterior_samples.numpy(), labels=keys, figsize=(8, 12))
for i in range(K):
    ax[i].axvline(THETA_test[perm_idx, i], color='r', linestyle='--')
plt.savefig(PATH + 'pdf/marginal.pdf', dpi=300)

plt.close('all')

if args.quality:

    # training performance on test set
    num_tests = BETA_test.shape[0]

    num_samples = 1000

    accuracy    = np.zeros((num_tests, K))
    uncertainty = np.zeros((num_tests, K))

    for i in range(num_tests):
        posterior.set_default_x(BETA_test[i])
        posterior_samples = posterior.sample((num_samples,))

        # for each test sample compute the accuracy and uncertainty of the posterior

        # accuracy
        accuracy[i, :] = np.mean(np.abs(posterior_samples.numpy() - THETA_test[i]), axis=0)

        # uncertainty
        uncertainty[i, :] = np.std(posterior_samples.numpy(), axis=0)

    # plot accuracy and uncertainty
    plt.figure(figsize=(10, 10))
    plt.suptitle('Accuracy')
    for i in range(K):
        plt.subplot(4, 2, i+1)
        plt.hist(accuracy[:, i], bins=100, color='k')
        plt.xlim(0, 1)
        plt.title(keys[i])
        plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig(PATH + 'pdf/accuracy.pdf')

    plt.figure(figsize=(10, 10))
    plt.suptitle('Uncertainty')
    for i in range(K):
        plt.subplot(4, 2, i+1)
        plt.hist(uncertainty[:, i], bins=100, color='k')
        plt.xlim(0, 1)
        plt.title(keys[i])
        plt.xlabel('Uncertainty')
    plt.tight_layout()
    plt.savefig(PATH + 'pdf/uncertainty.pdf')


if args.empirical:
    # load empirical data
    x_amb_LH_V1 = np.load('data/pizzuti/processed_by_Kris/sub-01_depth_vs_Amb_LH_V1.npy', allow_pickle=True)
    x_amb_RH_V1 = np.load('data/pizzuti/processed_by_Kris/sub-01_depth_vs_Amb_RH_V1.npy', allow_pickle=True)
    x_phy_LH_V1 = np.load('data/pizzuti/processed_by_Kris/sub-01_depth_vs_Phy_LH_V1.npy', allow_pickle=True)
    x_phy_RH_V1 = np.load('data/pizzuti/processed_by_Kris/sub-01_depth_vs_Phy_RH_V1.npy', allow_pickle=True)
    x = [x_amb_LH_V1, x_amb_RH_V1, x_phy_LH_V1, x_phy_RH_V1]
    labels = ['Amb LH V1', 'Amb RH V1', 'Phy LH V1', 'Phy RH V1']

    for i in range(len(x)):
        posterior.set_default_x(x[i])
        posterior_samples = posterior.sample((num_samples,))

        keys = np.arange(K)

        # check if directory exists
        if not os.path.exists(PATH + 'pdf/empirical/'):
            os.makedirs(PATH + 'pdf/empirical/')

        fig, ax = pairplot(posterior_samples.numpy(), labels=keys, figsize=(20, 20))
        plt.savefig(PATH + 'pdf/empirical/pairplot_{}.pdf'.format(labels[i]), dpi=300)

        fig, ax = marginal_correlation(posterior_samples.numpy(), labels=keys, figsize=(10, 10))
        plt.savefig(PATH + 'pdf/empirical/marginal_correlation_{}.pdf'.format(labels), dpi=300)

        fig, ax = marginal(posterior_samples.numpy(), labels=keys, figsize=(8, 12))
        plt.savefig(PATH + 'pdf/empirical/marginal_{}.pdf'.format(labels[i]), dpi=300)

    plt.figure()
    peaks = np.zeros((len(x), K))
    for i in range(len(x)):
        posterior.set_default_x(x[i])
        posterior_samples = posterior.sample((num_samples,))
        # get the peak of where the histogram of the posterior samples is located
        for j in range(K):
            y = np.histogram(posterior_samples[:, j], bins=100)
            idx = np.argmax(y[0])
            peaks[i, j] = y[1][idx]
        plt.plot(keys, peaks[i, :]/peaks.max(axis=0), '-', label=labels[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATH + 'pdf/empirical/est_lam_neur_signal.pdf', dpi=300)

    plt.close('all')