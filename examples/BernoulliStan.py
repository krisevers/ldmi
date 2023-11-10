import numpy as np

import stan

import os
import sys
sys.path.insert(0, os.getcwd())

from pathlib import Path

# Define the Stan model
model = Path('ldmi/stanmodels/Bernoulli.stan').read_text()

data = {'N': 10, "y": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])}

posterior = stan.build(model, data=data, random_seed=1)

fit = posterior.sample(num_chains=4, num_samples=1000)

theta = fit['theta']

df = fit.to_frame()

import IPython; IPython.embed()

