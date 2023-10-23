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
parser.add_argument('--num_simulations', type=int, default=1, help='number of simulations to run')
parser.add_argument('--path', type=str, default='data/', help='path to save results')
parser.add_argument("--name", type=str, default='X_input', help='name of experiment')
args = parser.parse_args()

"""
Explore systemic parameters (i.e. parameters that are not related to neurovascular coupling or hemodynamics).

experiment sys_I: explore input currents to each population
"""

PATH = args.path

import os
# check if path exists
if not os.path.exists(PATH):
    os.makedirs(PATH)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

E = {'K': 12, 'area': 'V1', 'T': 40, 'onset': 10, 'offset': 20}   # experimental parameters

theta = np.transpose([
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L23E
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L23I
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L4E
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L4I
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L5E
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L5I
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L6E
    np.random.uniform(0,   200,    size=args.num_simulations),  # I_L6I
])

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
    Psi_i = F(E, theta_i)
    X_i = {"Psi": Psi_i, "theta": theta_i}
    X.append(X_i)

# gather results
X = comm.allgather(X)

if rank == 0:
    X = np.ravel(X)
    X = np.array(X)

    np.save(PATH + args.name + '.npy', X)
