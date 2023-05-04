import numpy as np
import pylab as plt

import stan 

import os
import sys
sys.path.insert(0, os.getcwd())

import IPython

from pathlib import Path

# sigma, beta, rho = 10, 2.667, 28

# load stan model
PATH = 'ldmi/stanmodels/Lorenz.stan'
lorenzstan = Path(PATH).read_text()

N = 100             # number of particles
dt = 0.01           # time step
t_sim = 10          # simulation time
T = int(t_sim/dt)   # number of time steps
data = {'N': N, 'T': T, 'dt': dt}

posterior = stan.build(lorenzstan, data=data, random_seed=1)

IPython.embed()