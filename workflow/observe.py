import numpy as np
import h5py
import pylab as plt

from maps.I2K import I2K

K = 13

PROB_K = I2K(K, 'macaque', 'V1', sigma=1)

# load current data
with h5py.File('data/test/data.h5', 'r') as hf:
    PSI     = hf['PSI'][:]
    THETA   = hf['THETA'][:]
    bounds  = hf['bounds'][:]
    keys    = hf['keys'][:]
hf.close()

# flatten probabilities along last two dimensions
PROB_K = PROB_K[:, 1:, 1:] # remove L1
PROB_K = PROB_K.reshape((K, -1))

PROB_K = PROB_K[:, :72]

PSI_rec = PSI[:,   :64]
PSI_cc  = PSI[:, 64:72]
PSI_th  = PSI[:, 72:  ]

PSI = np.concatenate((PSI_rec, PSI_cc, PSI_th), axis=1)

E_map = np.zeros((K, 72))
E_map[:, ::2] = 1
E_map[:, 64:] = 1


MAP = np.zeros((PSI.shape[0], K))
for i in range(PSI.shape[0]):
    MAP[i] = (PSI[i] @ (PROB_K * E_map).T)

# save laminar projection
with h5py.File('data/test/data.h5', 'a') as hf:
    hf.create_dataset('MAP',     data=MAP)
hf.close()
