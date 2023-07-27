import mpi4py.MPI as MPI
import tqdm as tqdm
import numpy as np

import simulators.DMF_model as DMF_model

from worker_DMF import worker

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, help='Set path to save data')
parser.add_argument('-n', '--nsim', type=int, default=100, help='Set number of simulations')
args = parser.parse_args()

PATH = 'data/' + args.path
num_simulations = args.nsim

# check if path exists, if not create it
import os
if not os.path.exists(PATH):
    os.makedirs(PATH)

"""
Explore relations between DMF model parameters and laminar BOLD response | Keep NVC and LBR parameters fixed.
"""

if rank == 0:
    print('\n')
    print("##################################################################")
    print("                 LAMINAR DYNAMIC MODEL INFERENCE                  ")
    print("                    Simulations for Inference                     ")
    print("                 Cortical Microcircuit Model: DMF                 ")
    print("                       by Kris Evers, 2023                        ")
    print("##################################################################")

# global simulation parameters
M = 8           # number of neuronal populations
K = 12          # number of cortical depths
T_sim = 50      # simulation time

# set DMF parameters
DMF_blacklist = ['K', 'dt', 'T']
DMF_paramlist = []
DMF_theta = []
for i in range(num_simulations):
    DMF_params = {'K': K, 'dt': 1e-4, 'T': T_sim}
    DMF_params = DMF_model.DMF_parameters(DMF_params)    # get default parameters
    # select parameters to explore
    DMF_params['P_L23E>L23E'] = np.random.uniform(0, 0.15)
    DMF_params['P_L4E>L23E']  = np.random.uniform(0, 0.15)
    DMF_params['P_L4E>L23I']  = np.random.uniform(0, 0.15)
    DMF_params['P_L4I>L23E']  = np.random.uniform(0, 0.15)
    DMF_params['P_L4I>L23I']  = np.random.uniform(0, 0.15)
    DMF_params['P_L23E>L4E']  = np.random.uniform(0, 0.15)
    DMF_params['P_L23E>L4I']  = np.random.uniform(0, 0.15)
    DMF_params['P_L23I>L4E']  = np.random.uniform(0, 0.15)
    DMF_params['P_L23I>L4I']  = np.random.uniform(0, 0.15)
    DMF_params['P_L23E>L5E']  = np.random.uniform(0, 0.15)
    DMF_params['P_L23E>L5I']  = np.random.uniform(0, 0.15)
    DMF_params['P_L5I>L23E']  = np.random.uniform(0, 0.15)
    DMF_params['P_L5I>L23I']  = np.random.uniform(0, 0.15)
    DMF_params['P_L6E>L4E']   = np.random.uniform(0, 0.15)
    DMF_params['P_L6E>L4I']   = np.random.uniform(0, 0.15)
    DMF_params['U_L23E']   = np.random.uniform(0,  15)
    DMF_params['U_L23I']   = np.random.uniform(0,  15)
    DMF_params['U_L4E']    = np.random.uniform(0,  15)
    DMF_params['U_L4I']    = np.random.uniform(0,  15)
    DMF_params['U_L5E']    = np.random.uniform(0,  15)
    DMF_params['U_L5I']    = np.random.uniform(0,  15)
    DMF_params['U_L6E']    = np.random.uniform(0,  15)
    DMF_params['U_L6I']    = np.random.uniform(0,  15)
    DMF_theta.append(DMF_params)
# save DMF exploration parameters 
DMF_paramlist = ['P_L23E>L23E', 'P_L4E>L23E', 'P_L4E>L23I', 'P_L4I>L23E', 'P_L4I>L23I', 'P_L23E>L4E', 'P_L23E>L4I', 'P_L23I>L4E', 'P_L23I>L4I', 'P_L23E>L5E', 'P_L23E>L5I', 'P_L5I>L23E', 'P_L5I>L23I', 'P_L6E>L4E', 'P_L6E>L4I', 'U_L23E', 'U_L23I', 'U_L4E', 'U_L4I', 'U_L5E', 'U_L5I', 'U_L6E', 'U_L6I']

if rank == 0: print("DMF parameters: {}".format(DMF_paramlist))

# divide parameters over workers
num_simulations_per_worker = int(num_simulations / size)
DMF_paras_per_worker = np.array_split(DMF_theta, comm.Get_size())
DMF_workers_params = DMF_paras_per_worker[rank]

if rank == 0:
    print('\n')
    print("#Workers: {} | #Simulations: {} | #Simulations per worker: {} ".format(size, num_simulations, num_simulations_per_worker))

# simulate
X = []
for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
    DMF = DMF_workers_params[i]
    X_i = worker(K=K, T=T_sim, DMF=DMF)
    X_i.update(DMF)

    X.append(X_i)

np.save(PATH + '/X_{}.npy'.format(rank), X)