import numpy as np

import mpi4py.MPI as MPI
import tqdm as tqdm

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


import sys
sys.path.append('../')

import argparse

from worker import F

import os 

parser = argparse.ArgumentParser(description='Generate dataset for combined neuronal- and hemodynamic-based model.')
parser.add_argument('--num_simulations', type=int, default=1,       help='number of simulations to run')
parser.add_argument('--path',            type=str, default='data/', help='path to save results')
parser.add_argument("--name",            type=str,                  help='name of experiment')
args = parser.parse_args()

"""
Explore systemic parameters (i.e. parameters that are not related to neurovascular coupling or hemodynamics).

experiment sys_I: explore input currents to each population
"""

PATH = args.path + args.name + '/'

import os
# check if path exists
if not os.path.exists(PATH):
    os.makedirs(PATH)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

E = {'K': 12, 'area': 'V1', 'T': 15, 'onset': 1, 'offset': 10}   # experimental parameters

bounds = [
    [0,   42],
    [0,   35],
    [0,   116],
    [0,   73],
    [0,   42],
    [0,   35],
    [0,   61],
    [0,   23],
]

# for each simulation set two random parameters to non-zero
theta = np.zeros((args.num_simulations, 8))
choice = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=(args.num_simulations, 2), replace=True)
# make sure that the two random parameters are not the same
for i in range(args.num_simulations):
    theta[i, choice[i, 0]] = np.random.uniform(bounds[choice[i, 0]][0], bounds[choice[i, 0]][1])
    theta[i, choice[i, 1]] = np.random.uniform(bounds[choice[i, 1]][0], bounds[choice[i, 1]][1])


params_per_worker = np.array_split(theta, comm.Get_size())
num_simulations_per_worker = int(args.num_simulations / size)
worker_params = params_per_worker[rank]

X = []

for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
    theta_i = {
        'I_L23E':       worker_params[i, 0],
        'I_L23I':       worker_params[i, 1],
        'I_L4E':        worker_params[i, 2],
        'I_L4I':        worker_params[i, 3],
        'I_L5E':        worker_params[i, 4],
        'I_L5I':        worker_params[i, 5],
        'I_L6E':        worker_params[i, 6],
        'I_L6I':        worker_params[i, 7],
    }
    Psi_i = F(E, theta_i, integrator='numba')
    X_i = {"Psi": Psi_i, "theta": theta_i}
    peak_amp_v = X_i['Psi']['peak_amp_v']
    peak_amp_k = X_i['Psi']['peak_amp_k']
    X_i['Psi'] = {
        'peak_amp_v': peak_amp_v,
        'peak_amp_k': peak_amp_k,
    }
    X.append(X_i)

    if rank == 0:
        print('{}/{}'.format(i+1, num_simulations_per_worker), end='\r')

# gather results
X = comm.allgather(X)

if rank == 0:
    X = np.ravel(X)
    X = np.array(X)

    np.save(PATH + 'X.npy', X)
    keys = np.array(list(X[0]['theta'].keys()))
    np.save(PATH + 'keys.npy', keys)
    np.save(PATH + 'bounds.npy', bounds)