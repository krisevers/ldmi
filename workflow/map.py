import numpy as np
import h5py
import pylab as plt

from maps.I2K import I2K

import argparse

"""
Map currents to synapses.
"""


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Compute laminar projection from I to K.')
    parser.add_argument('-p', '--path', type=str,   default='data', help='path to save results')
    parser.add_argument('--name',       type=str,   default='test', help='Name of data file')
    parser.add_argument('--K',          type=int,   default=21,     help='Number of cortical layers')
    parser.add_argument('--sigma',      type=float, default=1,      help='Standard deviation of Gaussian kernel')
    args = parser.parse_args()

    PATH = args.path + '/' + args.name + '/'

    K = args.K

    PROB_K = I2K(K, 'macaque', 'V1', sigma=args.sigma)

    # load current data
    print('Loading data...')
    with h5py.File(PATH + 'data.h5', 'r') as hf:
        CURRENT     = hf['CURRENT'][:]
        RATE        = hf['RATE'][:]
        THETA       = hf['THETA'][:]
        BASELINE    = hf['BASELINE'][:]
        bounds      = hf['bounds'][:]
        keys        = hf['keys'][:]
    hf.close()

    # flatten probabilities along last two dimensions
    PROB_K = np.array([np.concatenate((np.ravel(PROB_K[k, :, :8]), PROB_K[k, :, 8], PROB_K[k, :, 9])) for k in range(K)])

    print('Selecting excitatory synapses...')
    E_map = np.zeros((K, 80))
    E_map[:, ::2] = 1
    E_map[:, 64:] = 1

    print('Computing laminar projection of currents to synapses...')
    MAP = np.zeros((CURRENT.shape[0], K))
    for i in range(CURRENT.shape[0]):
        MAP[i] = (CURRENT[i] @ (PROB_K * E_map).T)

    print('Computing baseline currents...')
    BASELINE = BASELINE @ (PROB_K * E_map).T

    print('Saving data...')
    hf = h5py.File(PATH + 'map.h5', 'w')
    hf.create_dataset('CURRENT',  data=MAP)
    hf.create_dataset('BASELINE', data=BASELINE)
    hf.close()