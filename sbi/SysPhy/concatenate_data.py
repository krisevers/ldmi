import numpy as np 

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
bounds = []
keys   = []
num_choices = []
X = []
for source in args.sources:
    path_source = args.path + args.name + '/' + source + '/'
    bounds.append(np.load(path_source + '/bounds.npy'))
    keys.append(np.load(path_source + '/keys.npy'))
    num_choices.append(np.load(path_source + '/num_choices.npy'))
    X.append(np.load(path_source + '/X.npy', allow_pickle=True))

# concatenate data
print('Concatenating data...')
bounds = bounds[0]
keys   = np.unique(np.concatenate(keys))
num_choices = np.concatenate(num_choices)
X = np.concatenate(X)

# save data
print('Saving data...')
np.save(PATH + '/bounds.npy', bounds)
np.save(PATH + '/keys.npy', keys)
np.save(PATH + '/num_choices.npy', num_choices)
np.save(PATH + '/X.npy', X, allow_pickle=True)