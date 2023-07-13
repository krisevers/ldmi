import mpi4py.MPI as MPI
import tqdm as tqdm
import numpy as np

import simulators.DCM_model as DCM_model
import simulators.NVC_model as NVC_model
import simulators.LBR_model as LBR_model

from worker import worker

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, help='Set path to save data')
parser.add_argument('-m', '--models', nargs='+', default='DCM', help='Set models to explore parameters')
parser.add_argument('-n', '--nsim', type=int, default=100, help='Set number of simulations')
args = parser.parse_args()

PATH = 'data/' + args.path
model_name = args.models
num_simulations = args.nsim

# check if path exists, if not create it
import os
if not os.path.exists(PATH):
    os.makedirs(PATH)

"""
Explore relations between NVC and LBR model parameters | Keep DCM parameters fixed.
"""

if rank == 0:
    print('\n')
    print("##################################################################")
    print("                 LAMINAR DYNAMIC MODEL INFERENCE                  ")
    print("                    Simulations for Inference                     ")
    print("                       by Kris Evers, 2023                        ")
    print("##################################################################")

# global simulation parameters
K = 12          # number of cortical depths
T_sim = 50      # simulation time

# set DCM parameters
if 'DCM' in model_name:
    DCM_blacklist = ['K', 'dt', 'T']
    DCM_paramlist = []
    DCM_theta = []
    for i in range(num_simulations):
        DCM_params = {'K': K, 'dt': 0.001, 'T': T_sim}
        DCM_params = DCM_model.DCM_parameters(K, DCM_params)
        DCM_params['U_1']   = np.random.uniform(0,  1)
        DCM_params['U_2']   = np.random.uniform(0,  1)
        DCM_params['U_3']   = np.random.uniform(0,  1)
        DCM_params['dur']   = np.random.uniform(1,  10)
        DCM_params['sigma'] = np.random.uniform(-1, -10)
        DCM_params['mu']    = np.random.uniform(-1, -10)
        DCM_params['lambda_'] = np.random.uniform(0, 1)
        DCM_theta.append(DCM_params)
    DCM_paramlist = ['U_1', 'U_2', 'U_3', 'dur', 'sigma', 'mu', 'lambda_']

    if rank == 0: print("DCM parameters: {}".format(DCM_paramlist))

# set NVC parameters
if 'NVC' in model_name:
    NVC_blacklist = ['K', 'dt', 'T']
    NVC_paramlist = []
    NVC_theta = []
    for i in range(num_simulations):
        NVC_params = {'K': K, 'dt': 0.001, 'T': T_sim}
        NVC_params = NVC_model.NVC_parameters(K, NVC_params)
        NVC_params['c1'] = np.random.uniform(0.0, 1.0)
        NVC_params['c2'] = np.random.uniform(0.0, 2.0)
        NVC_params['c3'] = np.random.uniform(0.0, 1.0)
        NVC_theta.append(NVC_params)
    NVC_paramlist = ['c1', 'c2', 'c3']

    if rank == 0: print("NVC parameters: {}".format(NVC_paramlist))

# set LBR parameters
if 'LBR' in model_name:
    LBR_blacklist = ['x_v', 'x_d', 'l', 'K', 'dt', 'T']
    LBR_paramlist = []
    LBR_theta = []
    for i in range(num_simulations):
        LBR_params = {'K': K, 'dt': 0.01, 'T': T_sim}
        LBR_params = LBR_model.LBR_parameters(K, LBR_params)
        LBR_params['E0v']       = np.random.uniform(0.25, 0.5)
        LBR_params['E0d']       = np.random.uniform(0.25, 0.5)
        LBR_params['E0p']       = np.random.uniform(0.25, 0.5)
        LBR_params['alpha_v']   = np.random.uniform(0.0,  0.6)
        LBR_params['alpha_d']   = np.random.uniform(0.0,  0.6)
        LBR_params['alpha_p']   = np.random.uniform(0.0,  0.6)
        LBR_params['Hct_v']     = np.random.uniform(0.32, 0.38)
        LBR_params['Hct_d']     = np.random.uniform(0.35, 0.41)
        LBR_params['Hct_p']     = np.random.uniform(0.38, 0.44)
        LBR_params['rho_v']     = 0.95 - LBR_params['Hct_v'] * 0.22 # no exploratory parameter, dependent on Hct_v
        LBR_params['rho_d']     = 0.95 - LBR_params['Hct_d'] * 0.22 # no exploratory parameter, dependent on Hct_d
        LBR_params['rho_p']     = 0.95 - LBR_params['Hct_p'] * 0.22 # no exploratory parameter, dependent on Hct_p
        LBR_params['R2s_t']     = np.random.uniform(30, 45)
        LBR_params['R2s_v']     = np.random.uniform(60, 250)
        LBR_params['R2s_d']     = np.random.uniform(60, 250)
        LBR_params['R2s_p']     = np.random.uniform(60, 250)
        LBR_params['tau_v_in']  = np.random.uniform(0, 100)
        LBR_params['tau_v_de']  = np.random.uniform(0, 100)
        LBR_params['tau_d_in']  = np.random.uniform(0, 100)
        LBR_params['tau_d_de']  = np.random.uniform(0, 100)
        LBR_params['tau_p_in']  = np.random.uniform(0, 100)
        LBR_params['tau_p_de']  = np.random.uniform(0, 100)
        LBR_params['t0v']       = np.random.uniform(0.5, 2)
        LBR_params['V0t']       = np.random.uniform(1, 5)
        LBR_params['V0_p']      = np.random.uniform(1, 10)
        LBR_params['w_v']       = np.random.uniform(0.25, 0.75) 
        LBR_theta.append(LBR_params)
    LBR_paramlist = [
                    'E0v', 'E0d', 'E0p', 
                    'alpha_v', 'alpha_d', 'alpha_p',
                    'Hct_v', 'Hct_d', 'Hct_p',
                    'R2s_t', 'R2s_v', 'R2s_d', 'R2s_p', 
                    'tau_v_in', 'tau_v_de',
                    'tau_d_in', 'tau_d_de',
                    'tau_p_in', 'tau_p_de', 
                    't0v', 'V0t', 'V0_p', 'w_v']
    
    if rank == 0: print("LBR parameters: {}".format(LBR_paramlist))

if rank == 0:
    print('\n')
    print("##################################################################")   
    print("Running {} simulations | Explore parameters for {} models".format(num_simulations, model_name))
    print("##################################################################")
    
# divide parameters over workers
num_simulations_per_worker = int(num_simulations / size)
if 'DCM' in model_name:
    DCM_paras_per_worker = np.array_split(DCM_theta, comm.Get_size())
    DCM_workers_params = DCM_paras_per_worker[rank]
if 'NVC' in model_name:
    NVC_paras_per_worker = np.array_split(NVC_theta, comm.Get_size())
    NVC_workers_params = NVC_paras_per_worker[rank]
if 'LBR' in model_name:
    LBR_paras_per_worker = np.array_split(LBR_theta, comm.Get_size())
    LBR_workers_params = LBR_paras_per_worker[rank]

if rank == 0:
    print('\n')
    print("#Workers: {} | #Simulations: {} | #Simulations per worker: {} ".format(size, num_simulations, num_simulations_per_worker))

# simulate
X = []
for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
    DCM = None
    NVC = None
    LBR = None
    if 'DCM' in model_name: DCM = DCM_workers_params[i]
    if 'NVC' in model_name: NVC = NVC_workers_params[i]
    if 'LBR' in model_name: LBR = LBR_workers_params[i]
    X_i = worker(K=K, T=T_sim, DCM=DCM, NVC=NVC, LBR=LBR)
    if 'DCM' in model_name: X_i.update(DCM)
    if 'NVC' in model_name: X_i.update(NVC)
    if 'LBR' in model_name: X_i.update(LBR)

    X.append(X_i)

np.save(PATH + '/X_{}.npy'.format(rank), X)

# all_X = comm.allgather(X)

# if rank == 0:
#     print('\n')
#     print("##################################################################")
#     print("Saving data...")

#     import IPython
#     IPython.embed()

#     all_X = np.concatenate(all_X)

#     # save data
#     np.save(PATH + '/X.npy', all_X)

#     if 'DCM' in model_name:
#         np.save(PATH + '/DCM_paramlist.npy', DCM_paramlist)
#     if 'NVC' in model_name:
#         np.save(PATH + '/NVC_paramlist.npy', NVC_paramlist)
#     if 'LBR' in model_name:
#         np.save(PATH + '/LBR_paramlist.npy', LBR_paramlist)

#     print("Done!")
#     print("##################################################################")

# TODO: reduce data load by saving compressed data (e.g. np.savez_compressed, pandas dataframe, hdf5).
# TODO: get example empirical BOLD response to test inference (is the NVC + LBR model able to reproduce the empirical BOLD response?)
    # TODO: extract summary statistics from empirical BOLD response
# TODO: add connection parameters in DCM model
# TODO: allow experimental design in DCM model (e.g. block design, event-related design, etc.)
# TODO: allow user to design summary statistics (e.g. peak amplitude, peak latency, peak area, undershoot amplitude, undershoot latency, undershoot area, etc.)