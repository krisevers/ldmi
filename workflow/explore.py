import numpy as np
import h5py

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../')

import argparse

parser = argparse.ArgumentParser(description='Explore dataset')
parser.add_argument('-p', '--path',     type=str, default='data',   help='path to load results from')
parser.add_argument(      '--name',     type=str,                   help='name of run')
args = parser.parse_args()

"""
Explore dataset.
"""

PATH = args.path + '/' + args.name + '/'

import os
# check if path exists
if not os.path.exists(PATH):
    raise ValueError('Path does not exist.')

# load dataset
hf = h5py.File(PATH + 'data.h5', 'r')

keys = hf.keys()

PSI     = np.array(hf.get('PSI'))
THETA   = np.array(hf.get('THETA'))
bounds  = np.array(hf.get('bounds'))

hf.close()


import pylab as plt

M = 8

# check if directory exists
if not os.path.exists(PATH + 'pdf/'):
    os.makedirs(PATH + 'pdf/')

# plot distributions of THETA
populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
plt.figure(figsize=(10, 10))
plt.suptitle('Parameters')
for i in range(8):
    plt.subplot(4, 2, i+1)
    plt.hist(THETA[:, i], bins=100)
    plt.title(populations[i])
    plt.xlabel(r'$\Theta$')
plt.tight_layout()
plt.savefig(PATH + 'pdf/THETA.pdf')
plt.close()

# plot distributions of PSI (recurrent connections)
connections = ['L23E-L23E', 'L23E-L23I', 'L23E-L4E', 'L23E-L4I', 'L23E-L5E', 'L23E-L5I', 'L23E-L6E', 'L23E-L6I',
               'L23I-L23E', 'L23I-L23I', 'L23I-L4E', 'L23I-L4I', 'L23I-L5E', 'L23I-L5I', 'L23I-L6E', 'L23I-L6I',
               'L4E-L23E',  'L4E-L23I',  'L4E-L4E',  'L4E-L4I',  'L4E-L5E',  'L4E-L5I',  'L4E-L6E',  'L4E-L6I',
               'L4I-L23E',  'L4I-L23I',  'L4I-L4E',  'L4I-L4I',  'L4I-L5E',  'L4I-L5I',  'L4I-L6E',  'L4I-L6I',
               'L5E-L23E',  'L5E-L23I',  'L5E-L4E',  'L5E-L4I',  'L5E-L5E',  'L5E-L5I',  'L5E-L6E',  'L5E-L6I',
               'L5I-L23E',  'L5I-L23I',  'L5I-L4E',  'L5I-L4I',  'L5I-L5E',  'L5I-L5I',  'L5I-L6E',  'L5I-L6I',
               'L6E-L23E',  'L6E-L23I',  'L6E-L4E',  'L6E-L4I',  'L6E-L5E',  'L6E-L5I',  'L6E-L6E',  'L6E-L6I',
               'L6I-L23E',  'L6I-L23I',  'L6I-L4E',  'L6I-L4I',  'L6I-L5E',  'L6I-L5I',  'L6I-L6E',  'L6I-L6I']
plt.figure(figsize=(10, 10))
plt.suptitle('Recurrent input')
for i in range(M**2):
    plt.subplot(8, 8, i+1)
    plt.hist(PSI[:, i], bins=100)
    plt.title(connections[i])
    plt.xlabel(r'$\Psi$')
plt.tight_layout()
plt.savefig(PATH + 'pdf/PSI.pdf')
plt.close()

# plot distributions of PSI (external connections)
plt.figure(figsize=(10, 10))
plt.suptitle('External input')
for i in range(M):
    plt.subplot(4, 2, i+1)
    plt.hist(PSI[:, M**2+i], bins=100)
    plt.title(populations[i])
    plt.xlabel(r'$\Psi$')
plt.tight_layout()
plt.savefig(PATH + 'pdf/PSI_ext.pdf')
plt.close()


import IPython; IPython.embed()
