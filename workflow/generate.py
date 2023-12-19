import numpy as np
import h5py

from models.DMF import DMF

import mpi4py.MPI as MPI
import tqdm as tqdm

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../')

import argparse

parser = argparse.ArgumentParser(description='Generate dataset for neuronal response to external input microcircuit model.')
parser.add_argument('-n', '--num_sims', type=int, default=1,            help='number of simulations')
parser.add_argument('-p', '--path',     type=str, default='data',       help='path to save results')
parser.add_argument(      '--name',     type=str,                       help='name of run')
parser.add_argument('-a', '--area',     type=str, default='unspecific', help='area to simulate')
args = parser.parse_args()

"""
Generate dataset of neuronal response to external input in the microcircuit model.
"""

PATH = args.path + '/' + args.name + '/'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# wait for rank 0 to create file
comm.Barrier()

dt = 1e-4
t_sim = 1
T = int(t_sim / dt)
M = 8

keys   = ['IL23E', 'IL23I', 'IL4E', 'IL4I', 'IL5E', 'IL5I', 'IL6E', 'IL6I']
bounds = [
    [0,     200 ], # IL23E
    [0,     200 ], # IL23I
    [0,     200 ], # IL4E
    [0,     200 ], # IL4I
    [0,     200 ], # IL5E
    [0,     200 ], # IL5I
    [0,     200 ], # IL6E
    [0,     200 ], # IL6I
]
if args.area == 'unspecific':
    bounds = [
        [0,     200 ], # IL23E
        [0,     200 ], # IL23I
        [0,     200 ], # IL4E
        [0,     200 ], # IL4I
        [0,     200 ], # IL5E
        [0,     200 ], # IL5I
        [0,     200 ], # IL6E
        [0,     200 ], # IL6I
    [30774, 64447   ], # NL23E
    [8680,  18178   ], # NL23I
    [10645, 70387   ], # NL4E
    [2661,  17597   ], # NL4I
    [7681,  20740   ], # NL5E
    [1686,  4554    ], # NL5I
    [7864,  34601   ], # NL6E
    [1610,  7086    ], # NL6I
    ]
    keys += ['NL23E', 'NL23I', 'NL4E', 'NL4I', 'NL5E', 'NL5I', 'NL6E', 'NL6I']

num_Iext = M # number of external inputs
num_N    = M # number of populations
if args.area == 'unspecific':
    num_parameters = num_Iext + num_N # 8 population sizes + 8 external inputs
else:
    num_parameters = num_Iext
theta = np.zeros((args.num_sims, num_parameters))
# fill theta with random values for each population within the bounds
for i in range(num_parameters):
    theta[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], args.num_sims)

params_per_worker = np.array_split(theta, comm.Get_size())
num_simulations_per_worker = int(args.num_sims / size)
worker_params = params_per_worker[rank]

DATA = []

for i in tqdm.tqdm(range(num_simulations_per_worker)):
    dt = 1e-4
    I_ext = np.zeros((T, M))
    I_ext[int(0.6/dt):int(0.9/dt), :] = worker_params[i, :num_Iext]

    if args.area == 'unspecific':
        N = worker_params[i, num_Iext:]
        area = None
    else:
        N = None
        area = args.area

    _, _, I, F = DMF(I_th=I_ext, I_cc=np.zeros((T, M)), N=N, area=area)

    CURRENT = I[int(0.7/dt)]
    RATE    = F[int(0.7/dt)]
    CURRENT_BASE = I[int(0.5/dt)]
    RATE_BASE    = F[int(0.5/dt)]
    THETA = worker_params[i]

    DATA.append({'THETA':           THETA,
                 'CURRENT':         CURRENT,
                 'RATE':            RATE, 
                 'CURRENT_BASE':    CURRENT_BASE,
                 'RATE_BASE':       RATE_BASE})

DATA = comm.allgather(DATA)

if rank == 0:
    DATA     = np.ravel(DATA)
    DATA     = np.array(DATA)

    THETA    = np.array([DATA[i]['THETA']       for i in range(len(DATA))])
    CURRENT  = np.array([DATA[i]['CURRENT']     for i in range(len(DATA))])
    RATE     = np.array([DATA[i]['RATE']        for i in range(len(DATA))])
    CURRENT_BASE = DATA[0]['CURRENT_BASE']  # baseline is the same for all simulations
    RATE_BASE    = DATA[0]['RATE_BASE']     # baseline is the same for all simulations

    import os
    # check if path exists
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    hf = h5py.File(PATH + 'dmf.h5', 'w')
    hf.create_dataset('CURRENT',        data=CURRENT)
    hf.create_dataset('RATE',           data=RATE)
    hf.create_dataset('THETA',          data=THETA)
    hf.create_dataset('CURRENT_BASE',   data=CURRENT_BASE)
    hf.create_dataset('RATE_BASE',      data=RATE_BASE)
    hf.create_dataset('bounds',         data=bounds)
    hf.create_dataset('keys',           data=keys)
    hf.close()