import mpi4py.MPI as MPI
import tqdm as tqdm
import numpy as np

import sys
sys.path.append('../')

from worker_DMF import worker
from utils import create_theta

import argparse

parser = argparse.ArgumentParser(description='Run DMF simulations.')
parser.add_argument('--num_simulations', type=int, default=1, help='number of simulations to run')
parser.add_argument('--path', type=str, default='data/', help='path to save results')
parser.add_argument('--verbose', type=bool, default=False, help='verbose')
args = parser.parse_args()

PATH = args.path

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

values = []
num_simulations = args.num_simulations

# eperimental parameters (keep fixed across simulations)
K = 12          # number of cortical depths
T_sim = 30      # simulation time
L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I = 0, 1, 2, 3, 4, 5, 6, 7
nu = 15                                             # Hz
S_th = 902                                          # number of thalamic inputs
P_th = np.array([0.0983, 0.0619, 0.0512, 0.0196])   # connection probability thalamic inputs
E = {'T': T_sim, 'TR': 2, 'K': K, 
        'stimulations': 
                           [{'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[0],  'target': [L4E]},
                            {'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[1],  'target': [L4I]},
                            {'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[2],  'target': [L6E]},
                            {'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[3],  'target': [L6I]}]
        }

theta = create_theta(num_simulations, components=['NVC', 'LBR'], parameters=[['c1', 'c2', 'c3'],
                                                                             ['V0t', 'V0t_p', 'Hct_v', 'Hct_d', 'Hct_p', 'alpha_v', 'alpha_d', 'alpha_p', 'B0', 'rho_t', 'rho_tp', 'R2s_t', 'R2s_v', 'R2s_d', 'R2s_p']])

params_per_worker = np.array_split(theta, comm.Get_size())

num_simulations_per_worker = int(num_simulations / size)
worker_params = params_per_worker[rank]

X = []
for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
    theta_i = worker_params[i]
    X_i = worker(E, O={}, theta=theta_i)
    X_i.update(theta_i)
    X_i['peak_Ampl'] = X_i['peak_Ampl'][0]
    X_i['unde_Ampl'] = X_i['unde_Ampl'][0]
    X_i['tota_Area'] = X_i['tota_Area'][0]
    X.append(X_i)

# gather results
X = comm.allgather(X)

if rank == 0:
    X = np.ravel(X)
    X = np.array(X)

    import os
    if os.path.exists(PATH):
        print('Path already exists. Overwriting.')
    else:
        os.makedirs(PATH)

    np.save(PATH + 'X.npy', X)

    if args.verbose:
        print('Saved results to ' + PATH + 'X.npy')