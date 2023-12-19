import numpy as np
import h5py
import json

from maps.I2K import I2K

import argparse
from mpi4py import MPI

"""
Using a protocol, for each simulation, generate a timeseries of currents and thetas.
Then use hemodynamic models (NVC and LBR) to obtain the laminar BOLD response.
Save the results in the data file.
"""

from models.NVC import NVC      # neurovascular coupling model  (function)
from models.LBR import LBR      # laminar BOLD response model   (class)
from models.HRF import HRF      # hemodynamic response function (function)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Build all possible combinations of protocol conditions.')
    parser.add_argument('-p', '--path', type=str,   default='data',         help='path to save results')
    parser.add_argument('--name',       type=str,   default='test',         help='Name of data file')
    parser.add_argument('--show',       action='store_true',                help='Show protocol')
    parser.add_argument('--theta',      action='store_true',                help='free NVC and LBR parameters')
    args = parser.parse_args()

    PATH = args.path + '/' + args.name + '/'

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print('Loading data...')
    # load THETA
    hf = h5py.File(PATH + 'dmf.h5', 'r')
    THETA       = hf['THETA'][:]        # parameters (num_sims x num_params)
    hf.close()
    # load currents
    hf = h5py.File(PATH + 'map.h5', 'r')
    CURRENT     = hf['CURRENT'][:]      # laminar projection of currents to synapses (num_sims x K)
    BASELINE    = hf['BASELINE'][:]     # baseline currents (K)
    hf.close()
    # load protocol data
    hf = h5py.File(PATH + 'protocol.h5', 'r')
    timesteps   = hf['timesteps'][:]    # time steps (num_timesteps)
    protocol    = hf['protocol'][:]     # protocol (num_timesteps)
    hf.close()

    if rank == 0:
        print('Set parameters...')
    num_conditions  = len(np.unique(protocol)) - 1  # number of conditions (0 is baselline)
    num_timesteps   = int(len(timesteps))           # number of timesteps
    num_sims        = int(CURRENT.shape[0])         # number of simulations
    K               = int(CURRENT.shape[1])         # number of cortical depths

    lbr = LBR(K)
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

    # renormalize all currents such that the median of the response is 1
    median = np.median(CURRENT)
    CURR = (CURRENT - BASELINE) / median


    if rank == 0:
        print('Compute laminar BOLD responses and obtain betas...')

    # Distribute simulations across processes
    sim_per_process = num_sims // size
    start_sim = rank * sim_per_process
    end_sim = start_sim + sim_per_process

    BETA = np.zeros((num_sims, K))


    for s in range(start_sim, end_sim):
        print('Process %d: Simulation %d' % (rank, s), end='\r')
        curr = np.zeros((num_timesteps, K))
        for c in np.unique(protocol):
            if c == 0: continue # skip baseline
            else:
                # find indices of condition
                idx = np.where(protocol == c)[0]
                # generate timeseries
                for t in idx:
                    curr[t] = CURR[s]  # subtract baseline
        
        # compute neurovascular response
        if args.theta:
            F = NVC(curr, theta=THETA[s])
        else:
            F = NVC(curr)   # F is a timeseries of cerebral blood flow (CBF) (num_timesteps x K)
        # compute BOLD response
        F = F[::int(lbr_dt/dt)]     # resample to match LBR timesteps
        if args.theta:
            B, _ = lbr.sim(F, K, theta=THETA[s], integrator='numba')
        else:
            B, _, _ = lbr.sim(F, K, integrator='numba')

        # compute betas
        beta = np.linalg.lstsq(X, B, rcond=None)[0]
        BETA[s] = beta[0]

        del curr, F, B   # free memory

    # Gather results from all processes
    all_BETA  = comm.allgather(BETA)

    if rank == 0:

        # stack results from all processes
        BETA  = np.array(np.sum(all_BETA, axis=0))

        print('Saving data...')
        hf = h5py.File(PATH + 'hemo.h5', 'w')
        hf.create_dataset('BETA', data=BETA)
        hf.create_dataset('THETA', data=THETA)
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

            plt.figure()
            for i in range(8):
                plt.subplot(2, 4, i+1)
                plt.scatter(THETA[:, i], BETA)
                plt.title('Parameter {}'.format(i))
                plt.xlabel('Parameter value')
                plt.ylabel('BETA')
            plt.tight_layout()
            plt.savefig(PATH + 'pdf/THETA_BETA.pdf', dpi=300)
            
