PATH = 'data/'

import mpi4py.MPI as MPI
import tqdm as tqdm
import numpy as np

from worker_DMF import worker
from utils import create_theta

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

values = []
num_simulations = size

# eperimental parameters (keep fixed across simulations)
K = 12          # number of cortical depths
T_sim = 80      # simulation time
L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I = 0, 1, 2, 3, 4, 5, 6, 7
E = {'T': T_sim, 'TR': 2, 'K': K, 
        'stimulations': 
                        [{'onset': 5,  'duration': 10, 'amplitude': 10,  'target': [L23E]},
                            {'onset': 25, 'duration': 10, 'amplitude': 10,  'target': [L4E]},
                            {'onset': 45, 'duration': 10, 'amplitude': 10,  'target': [L5E]},
                            {'onset': 65, 'duration': 10, 'amplitude': 10,  'target': [L6E]}]
        }

theta = create_theta(num_simulations, components=['NVC', 'LBR'], parameters=[['c1', 'c2', 'c3'],['V0t', 'V0t_p']])

params_per_worker = np.array_split(theta, comm.Get_size())

num_simulations_per_worker = int(num_simulations / size)
worker_params = params_per_worker[rank]

X = []
for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
    theta_i = worker_params[i]
    X_i = worker(E, O={}, theta=theta_i)
    X_i.update(theta_i)
    X.append(X_i)

# gather results
X = comm.allgather(X)

if rank == 0:
    X = np.ravel(X)
    X = np.array(X)

    import pylab as plt
    plt.figure()
    plt.imshow(X[0]['lbr'].T, aspect='auto', interpolation='none')
    plt.xlabel('time (TR)')
    plt.ylabel('cortical depth (K)')
    plt.colorbar()
    plt.savefig('png/test/imshow_lbr_ex{}.png'.format(0))

    import IPython; IPython.embed()