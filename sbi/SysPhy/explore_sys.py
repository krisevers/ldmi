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
parser.add_argument("--name", type=str, help='name of experiment')
args = parser.parse_args()

"""
Explore systemic parameters (i.e. parameters that are not related to neurovascular coupling or hemodynamics).
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
    np.random.uniform(0.05,   .15,  size=args.num_simulations),  # P_L23EtoL23E
    np.random.uniform(0.02,   .09,  size=args.num_simulations),  # P_L4EtoL23E
    np.random.uniform(0.05,   .15,  size=args.num_simulations),  # P_L23EtoL5E
    np.random.uniform(0.05,   .15,  size=args.num_simulations),  # P_L23EtoL4I
    np.random.uniform(0.02,   .09,  size=args.num_simulations),  # P_L23EtoL6I
    np.random.uniform(0.02,   .10,  size=args.num_simulations),  # P_L5EtoL23I
    np.random.uniform(0.05,   .15,  size=args.num_simulations),  # P_L6E_L4I
    np.random.uniform(0.02,   .09,  size=args.num_simulations),  # P_L5E_L6E
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L23E
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L23I
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L4E
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L4I
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L5E
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L5I
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L6E
    np.random.uniform(0,   1500,    size=args.num_simulations),  # I_L6I
    # np.random.uniform(0.5, 1.5,     size=args.num_simulations),  # lam_E
    # np.random.uniform(0,   0.5,     size=args.num_simulations),  # lam_I
    # np.random.uniform(0.3, 2,       size=args.num_simulations),  # c1
    # np.random.uniform(0.3, 2,       size=args.num_simulations),  # c2
    # np.random.uniform(0.3, 2,       size=args.num_simulations),  # c3
    # np.random.uniform(0.1, 1,       size=args.num_simulations),  # E_0v
    # np.random.uniform(1,   4,       size=args.num_simulations),  # V_0t
    # np.random.uniform(0.01, 0.1,    size=args.num_simulations),  # TE
])

params_per_worker = np.array_split(theta, comm.Get_size())
num_simulations_per_worker = int(args.num_simulations / size)
worker_params = params_per_worker[rank]

X = []

for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
    theta_i = {
        'P_L23EtoL23E': worker_params[i, 0],
        'P_L4EtoL23E':  worker_params[i, 1],
        'P_L23EtoL5E':  worker_params[i, 2],
        'P_L23EtoL4I':  worker_params[i, 3],
        'P_L23EtoL6I':  worker_params[i, 4],
        'P_L5EtoL23I':  worker_params[i, 5],
        'P_L6E_L4I':    worker_params[i, 6],
        'P_L5E_L6E':    worker_params[i, 7],
        'I_L23E':       worker_params[i, 8],
        'I_L23I':       worker_params[i, 9],
        'I_L4E':        worker_params[i, 10],
        'I_L4I':        worker_params[i, 11],
        'I_L5E':        worker_params[i, 12],
        'I_L5I':        worker_params[i, 13],
        'I_L6E':        worker_params[i, 14],
        'I_L6I':        worker_params[i, 15],
        # 'lam_E':        worker_params[i, 16],
        # 'lam_I':        worker_params[i, 17],
        # 'c1':           worker_params[i, 18],
        # 'c2':           worker_params[i, 19],
        # 'c3':           worker_params[i, 20],
        # 'E_0v':         worker_params[i, 21],
        # 'V_0t':         worker_params[i, 22],
        # 'TE':           worker_params[i, 23]
    }
    Psi = F(E, theta_i)
    X_i = {"Psi": Psi, "theta": theta_i}
    X.append(X_i)

# gather results
X = comm.allgather(X)

if rank == 0:
    X = np.ravel(X)
    X = np.array(X)

    np.save(PATH + args.name + '.npy', X)

