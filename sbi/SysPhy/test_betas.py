import numpy as np
import pylab as plt

from worker import F

E = {'K': 12, 'area': 'V1', 'T': 40, 'onset': 5, 'offset': 10}   # experimental parameters

theta = {'I_L4E': 200, 'I_L4I': 180}

Psi = F(E, theta, mode='betas')

import IPython; IPython.embed()