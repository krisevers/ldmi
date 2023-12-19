import numpy as np
import h5py
import json

import pylab as plt

from maps.I2K import I2K

import argparse

"""
Setup protocol (i.e. time dependent conditions)
"""

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Build all possible combinations of protocol conditions.')
    parser.add_argument('-p', '--path', type=str,   default='data',         help='path to save results')
    parser.add_argument('--name',       type=str,   default='test',         help='Name of data file')
    parser.add_argument('--protocol',   type=str,   default='checkerboard', help='Name of protocol file')
    parser.add_argument('--show',       action='store_true',                help='Show protocol')
    args = parser.parse_args()

    PATH = args.path + '/' + args.name + '/'

    # load protocol data
    print('Loading protocol...')
    protocol = json.load(open('protocol.json'))

    # find protocol with the right name
    for p in protocol:
        if p['name'] == args.protocol:
            protocol = p
            break

    # build protocol
    print('Building protocol...')
    duration = protocol['duration']
    dt = 1e-4
    t = np.arange(0, duration, dt)
    x = np.zeros((len(t), 1))
    num_conditions = len(protocol['conditions'])
    cond_idx = np.arange(1, num_conditions+1)
    for c, cond in enumerate(protocol['conditions']):
        onset       = int(cond['onset'] / dt)      # onset in time steps
        duration    = int(cond['duration'] / dt)   # duration in time steps
        repetitions = int(cond['repetitions'])     # number of repetitions
        interval    = int(cond['interval'] / dt)   # interval between repetitions in time steps
        for r in range(repetitions):
            x[onset + r * (duration + interval) : onset + duration + r * (duration + interval)] = cond_idx[c]

    # save protocol
    print('Saving protocol...')
    hf = h5py.File(PATH + 'protocol.h5', 'w')
    hf.create_dataset('timesteps',  data=t)
    hf.create_dataset('protocol',   data=x)

    # show protocol
    if args.show:
        plt.plot(t, x, 'k')
        plt.xlabel('Time [s]')
        plt.ylabel('Condition')
        plt.savefig(PATH + 'pdf/protocol.pdf', dpi=300)
        plt.close('all')