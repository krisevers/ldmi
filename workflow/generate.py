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
parser.add_argument('-n', '--num_sims', type=int, default=1,        help='number of simulations')
parser.add_argument('-p', '--path',     type=str, default='data',   help='path to save results')
parser.add_argument(      '--name',     type=str,                   help='name of run')
parser.add_argument('-a', '--area',     type=str, default='V1',     help='area to simulate')
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
    [0,   200],
    [0,   200],
    [0,   200],
    [0,   200],
    [0,   200],
    [0,   200],
    [0,   200],
    [0,   200],
]

theta = np.zeros((args.num_sims, 8))
# fill theta with random values for each population within the bounds
for i in range(8):
    theta[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], args.num_sims)

params_per_worker = np.array_split(theta, comm.Get_size())
num_simulations_per_worker = int(args.num_sims / size)
worker_params = params_per_worker[rank]

DATA = []

for i in tqdm.tqdm(range(num_simulations_per_worker)):
    dt = 1e-4
    I_ext = np.zeros((T, M))
    I_ext[int(0.6/dt):int(0.9/dt), :] = worker_params[i]

    _, _, I = DMF(I_th=I_ext, I_cc=np.zeros((T, M)),  area=args.area)

    PSI = I[int(0.7/dt)]
    THETA = worker_params[i]
    BASELINE = I[int(0.5/dt)]

    DATA.append({'PSI': PSI, 'THETA': THETA, 'BASELINE': BASELINE})

DATA = comm.allgather(DATA)

if rank == 0:
    DATA     = np.ravel(DATA)
    DATA     = np.array(DATA)
    THETA    = np.array([DATA[i]['THETA']       for i in range(len(DATA))])
    PSI      = np.array([DATA[i]['PSI']         for i in range(len(DATA))])
    BASELINE = np.array([DATA[i]['BASELINE']    for i in range(len(DATA))])

    import os
    # check if path exists
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    hf = h5py.File(PATH + 'data.h5', 'w')
    hf.create_dataset('PSI',        data=PSI)
    hf.create_dataset('THETA',      data=THETA)
    hf.create_dataset('BASELINE',   data=BASELINE)
    hf.create_dataset('bounds',     data=bounds)
    hf.create_dataset('keys',       data=keys)
    hf.close()

