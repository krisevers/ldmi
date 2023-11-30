import numpy as np
import h5py

import argparse

from mpi4py import MPI

"""
Generate hemodynamics dataset based on random input data to just infer origin of laminar BOLD response.
"""

import sys
sys.path.append('../')
sys.path.append('../../')

from models.NVC import NVC      # neurovascular coupling model  (function)
from models.LBR import LBR      # laminar BOLD response model   (class)
from models.HRF import HRF      # hemodynamic response function (function)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Build all possible combinations of protocol conditions.')
    parser.add_argument('-p', '--path', type=str,       default='data',     help='path to save results')
    parser.add_argument('--name',       type=str,       default='hemo',     help='Name of data file')
    parser.add_argument('--show',       action='store_true',                help='Show protocol')
    parser.add_argument('-k', '--k',    type=int,       default=10,         help='Number of cortical depths')
    parser.add_argument('-n', '--num_sims', type=int,   default=1000,       help='Number of simulations')
    args = parser.parse_args()

    PATH = args.path + '/' + args.name + '/'

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    K = args.k
    num_sims = args.num_sims

    lbr = LBR(K)
    dt      = 1e-4
    lbr_dt  = 0.001
    num_timesteps = int(40/dt)
    num_lbr_timesteps = int(num_timesteps * dt / lbr_dt)

    if rank == 0:
        print('Convolve protocol with HRF...')
    hrf = HRF(np.arange(0, 10, lbr_dt))
    protocol = np.zeros((num_timesteps, 1))
    protocol[int(10/dt):int(20/dt)] = 1
    protocol_resampled = protocol[::int(lbr_dt/dt)][:, -1]
    X = np.convolve(protocol_resampled, hrf, mode='same')
    X /= np.max(X)              # normalize
    X = np.tile(X, (K, 1)).T    # repeat for each cortical depth


    if rank == 0:
        print('Compute laminar BOLD responses and obtain betas...')

    # Distribute simulations across processes
    sim_per_process = num_sims // size
    start_sim = rank * sim_per_process
    end_sim = start_sim + sim_per_process

    # Initialize data array
    DATA = np.zeros((sim_per_process, K, 2))

    comm.Barrier()


    for s in range(sim_per_process):
        print('Process %d: Simulation %d/%d     ' % (rank, s, sim_per_process), end='\r')
        I = np.zeros((num_timesteps, K))
        DATA[s, :, 0] = np.random.rand(K)
        I[int(10/dt):int(20/dt), :] = DATA[s, :, 0]
        # compute neurovascular response
        F = NVC(I)   # F is a timeseries of cerebral blood flow (CBF) (num_timesteps x K)
        # compute BOLD response
        F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps (num_lbr_timesteps x K)
        B, _, _ = lbr.sim(F, K, integrator='numba') # B is a timeseries of BOLD response (num_lbr_timesteps x K)

        # compute betas
        beta = np.linalg.lstsq(X, B, rcond=None)[0]
        DATA[s, :, 1] = beta[0]
        
        del I, F, B   # free memory

    # Gather results from all processes
    all_DATA  = comm.allgather(DATA)

    if rank == 0:
        # Concatenate results from all processes
        DATA = np.concatenate(all_DATA)

        THETA = DATA[:, :, 0]
        BETA  = DATA[:, :, 1]

        print('Saving data...')
        hf = h5py.File(PATH + 'hemo.h5', 'w')
        hf.create_dataset('BETA',  data=BETA)
        hf.create_dataset('THETA', data=THETA)
        hf.close()

        print("Saved data to {}".format(PATH + 'data.h5'))