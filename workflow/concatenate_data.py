import numpy as np 
import h5py

import os
import argparse

parser = argparse.ArgumentParser(description='Concatenate data from different simulations.')
parser.add_argument('--path',            type=str, default='data/', help='path to save results')
parser.add_argument('--name',            type=str,                  help='name of experiment')
# argument which lists the source folders where data to concatenate is stored
parser.add_argument('--sources',        type=str, nargs='+',       help='list of source folders')
args = parser.parse_args()

print('Concatenating data from simulations {}'.format(args.sources))

PATH = args.path + args.name + '/all/'

# check if path exists
if not os.path.exists(PATH):
    os.makedirs(PATH)

# load data from different simulations
print('Loading data...')
bounds      = []
keys        = []
PSI         = []
THETA       = []
BASELINE    = []
for source in args.sources:
    path_source = args.path + args.name + '/' + source + '/'
    hf = h5py.File(path_source + 'data.h5', 'r')
    bounds.append(np.array(hf.get('bounds')))
    keys.append(np.array(hf.get('keys')))
    PSI.append(np.array(hf.get('PSI')))
    THETA.append(np.array(hf.get('THETA')))
    BASELINE.append(np.array(hf.get('BASELINE')))
    hf.close()

# concatenate data
print('Concatenating data...')
bounds = bounds[0]
keys   = np.unique(np.concatenate(keys))
PSI    = np.concatenate(PSI)
THETA  = np.concatenate(THETA)
BASELINE = np.concatenate(BASELINE)

# save data
print('Saving data...')
hf = h5py.File(PATH + 'data.h5', 'w')
hf.create_dataset('bounds',     data=bounds)
hf.create_dataset('keys',       data=keys)
hf.create_dataset('PSI',        data=PSI)
hf.create_dataset('THETA',      data=THETA)
hf.create_dataset('BASELINE',   data=BASELINE)
hf.close()
