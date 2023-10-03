import mpi4py.MPI as MPI
import tqdm as tqdm
import numpy as np

import sys
sys.path.append('../')

import argparse

from PDCM import worker

import os

parser = argparse.ArgumentParser(description='Run PDCM simulations.')
parser.add_argument('--num_simulations', type=int, default=1, help='number of simulations to run')
parser.add_argument('--path', type=str, default='data/PDCM/', help='path to save results')
args = parser.parse_args()

PATH = args.path

import os
# check if path exists
if not os.path.exists(PATH):
    os.makedirs(PATH)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


theta = np.transpose([
    np.random.uniform(.3,    .9,    size=args.num_simulations),  # c1
    np.random.uniform( 1,     2,    size=args.num_simulations),  # c2
    np.random.uniform(.3,    .9,    size=args.num_simulations),  # c3
    np.random.uniform(1,      5,    size=args.num_simulations),  # tau_mtt
    np.random.uniform(0.1,    30,   size=args.num_simulations),  # tau_vs
    np.random.uniform(.1,    .5,    size=args.num_simulations),  # alpha
    np.random.uniform(.1,    .8,    size=args.num_simulations),  # E_0
    np.random.uniform(1,      10,   size=args.num_simulations),  # V_0
    np.random.uniform(.3390, .3967, size=args.num_simulations),  # eps
    np.random.uniform(10,     1000, size=args.num_simulations),  # rho_0
    np.random.uniform(40,     440,  size=args.num_simulations),  # nu_0
    np.random.uniform(.015,  .040,  size=args.num_simulations),  # TE
])

params_per_worker = np.array_split(theta, comm.Get_size())
num_simulations_per_worker = int(args.num_simulations / size)
worker_params = params_per_worker[rank]

X = []
for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
    theta_i = {
        'c1':      worker_params[i, 0],
        'c2':      worker_params[i, 1],
        'c3':      worker_params[i, 2],
        'tau_mtt': worker_params[i, 3],
        'tau_vs':  worker_params[i, 4],
        'alpha':   worker_params[i, 5],
        'E_0':     worker_params[i, 6],
        'V_0':     worker_params[i, 7],
        'eps':     worker_params[i, 8],
        'rho_0':   worker_params[i, 9],
        'nu_0':    worker_params[i, 10],
        'TE':      worker_params[i, 11],
    }
    _, _, _, _, _, _, stats = worker(theta=theta_i)
    X_i = {"stats": stats, "theta": theta_i}
    X.append(X_i)

# gather results
X = comm.allgather(X)

if rank == 0:
    X = np.ravel(X)
    X = np.array(X)

    if os.path.exists(PATH):
        print('Path already exists. Overwriting.')
    else:
        os.makedirs(PATH)

    np.save(PATH + 'X.npy', X)