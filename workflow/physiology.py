import numpy as np
import h5py
import json

import argparse
from mpi4py import MPI

"""
Using a protocol, for each simulation, generate a timeseries of currents and thetas.
Then use hemodynamic models (NVC and LBR) to obtain the laminar BOLD response.
Save the results in the data file.
"""

from models.NVC import NVC      # neurovascular coupling model (function)
from models.LBR import LBR      # laminar BOLD response model   (class)
from models.HRF import HRF      # hemodynamic response function (function)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Build all possible combinations of protocol conditions.')
    parser.add_argument('-p', '--path', type=str,   default='data',         help='path to save results')
    parser.add_argument('-n', '--n',    type=int,   default=100,            help='Number of simulations')
    parser.add_argument('--name',       type=str,   default='test',         help='Name of data file')
    parser.add_argument('--show',       action='store_true',                help='Show protocol')
    args = parser.parse_args()

    PATH = args.path + '/' + args.name + '/'

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print('Set parameters...')
    t_sim = 40.0   # simulation time (s)
    dt   = 1e-4    # timestep (s)
    num_timesteps   = int(t_sim/dt)     # number of timesteps
    num_sims        = args.n            # number of simulations
    K               = 21                # number of cortical depths

    protocol = np.zeros((num_timesteps, K))
    onset_time = 10.0
    duration = 10.0
    for k in range(K):
        protocol[int(onset_time/dt):int((onset_time+duration)/dt), k] = 1

    dt = 1e-4
    lbr_dt = 0.001
    num_lbr_timesteps = int(num_timesteps * dt / lbr_dt)

    if rank == 0:
        print('Convolve protocol with HRF...')
    hrf = HRF(np.arange(0, 10, lbr_dt))
    protocol_resampled = protocol[::int(lbr_dt/dt)][:, -1]
    X = np.convolve(protocol_resampled, hrf, mode='same')
    X /= np.max(X)              # normalize
    X = np.tile(X, (K, 1)).T    # repeat for each cortical depth


    if rank == 0:
        print('Compute laminar BOLD responses and obtain betas...')
    BETA = np.zeros((num_sims, K))
    THETA_ = np.zeros((num_sims, 8))

    # Distribute simulations across processes
    sim_per_process = num_sims // size
    start_sim = rank * sim_per_process
    end_sim = start_sim + sim_per_process


    num_parameters = 3  # number of parameters to estimate (c1, c2, c3)
    THETA = np.zeros((num_sims, num_parameters))

    bounds = np.array([[0.1, 2.0], [0.1, 2.0], [0.1, 2.0]])
    keys   = np.array(['c1', 'c2', 'c3'])


    for s in range(start_sim, end_sim):
        print('Process %d: Simulation %d' % (rank, s), end='\r')
        theta = {'c1': np.random.uniform(low=bounds[0, 0], high=bounds[0, 1]), 
                 'c2': np.random.uniform(low=bounds[1, 0], high=bounds[1, 1]),
                 'c3': np.random.uniform(low=bounds[2, 0], high=bounds[2, 1])
                 }
        
        # compute neurovascular response
        F = NVC(protocol, c1=theta['c1'], c2=theta['c2'], c3=theta['c3'])          # F is a timeseries of cerebral blood flow (CBF) (num_timesteps x K)
        F = F[::int(lbr_dt/dt)]                 # resample to match LBR timesteps

        lbr = LBR(K, theta=theta)
        B, _, _ = lbr.sim(F, K, integrator='numba')

        # compute betas
        beta = np.linalg.lstsq(X, B, rcond=None)[0]
        BETA[s] = beta[0]
        THETA[s] = np.array(list(theta.values()))

        del F, B   # free memory

    # Gather results from all processes
    all_BETA  = comm.allgather(BETA)
    all_THETA = comm.allgather(THETA)

    if rank == 0:
        # Concatenate results from all processes
        BETA  = np.sum(all_BETA,  axis=0)
        THETA = np.sum(all_THETA, axis=0)

        print('Saving data...')
        hf = h5py.File(PATH + 'data.h5', 'w')
        hf.create_dataset('BETA',  data=BETA)
        hf.create_dataset('THETA', data=THETA)
        # hf.create_dataset('bounds', data=bounds)
        # hf.create_dataset('keys', data=keys)
        hf.close()

        if args.show:

            import pylab as plt

            import os
            # check if path exists
            if not os.path.exists(PATH + 'pdf'):
                os.makedirs(PATH + 'pdf')

            plt.figure()
            plt.imshow(BETA.T, aspect='auto', cmap='PiYG', interpolation=None)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('BETA')
            plt.xlabel('Simulation')
            plt.ylabel('Cortical Depth')
            plt.savefig(PATH + 'pdf/BETA.pdf', dpi=300)

            plt.figure()
            plt.imshow(THETA.T, aspect='auto', cmap='PiYG', interpolation=None)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('THETA')
            plt.xlabel('Simulation')
            plt.ylabel('Parameters')
            plt.savefig(PATH + 'pdf/THETA.pdf', dpi=300)

            plt.close('all')
