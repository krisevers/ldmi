import numpy as np
import pylab as plt

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, help='Set path to obtain data')
args = parser.parse_args()

PATH = 'data/' + args.path

X = np.load(PATH + '/X.npy', allow_pickle=True)

num_simulations = X.shape[0]

"""
Obtain the summary statistics for each simulation
"""

sum_stat_keys = ['peak_Ampl', 'peak_Posi', 'peak_Area', 
                 'unde_Ampl', 'unde_Posi', 'unde_Area', 
                 'up_Slope', 'down_Slope']
np.save(PATH + '/sum_stat_keys.npy', sum_stat_keys)

blacklist_idx = []
for i in range(num_simulations):
    # check for nan values
    if np.isnan(X[i]['lbr']).any() or np.isnan(X[i]['cbf']).any() or np.isnan(X[i]['neuro']).any():
        print('Simulation {} contains NaN values.'.format(i))
        blacklist_idx.append(i)
        continue

    # remove outliers
    elif (abs(X[i]['lbr']) > 50).any() or (abs(X[i]['cbf']) > 50).any() or (abs(X[i]['neuro']) > 50).any():
        print('Simulation {} contains outliers.'.format(i))
        blacklist_idx.append(i)
        continue
    
    K = X[i]['K']
    T = X[i]['T']

    # get LBR statistics for each simulation
    peak_Posi   = np.zeros((K), dtype=int)
    peak_Ampl   = np.zeros((K))
    peak_Area   = np.zeros((K))
    unde_Posi   = np.zeros((K), dtype=int)
    unde_Ampl   = np.zeros((K))
    unde_Area   = np.zeros((K))
    up_Slope    = np.zeros((K))
    down_Slope  = np.zeros((K))
    for k in range(K):
        peak_Ampl[k] = np.max(X[i]['lbr'][:, k])                                                # response peak amplitude
        peak_Posi[k] = int(np.where(X[i]['lbr'][:, k] == peak_Ampl[k])[0][0])                   # response peak latency
        peak_Area[k] = np.sum(X[i]['lbr'][X[i]['lbr'][:, k] > 0, k])                            # response peak area
        unde_Ampl[k] = np.min(X[i]['lbr'][peak_Posi[k]:, k])                                    # undershoot amplitude
        unde_Posi[k] = int(np.where(X[i]['lbr'][:, k] == unde_Ampl[k])[0][0])                   # undershoot latency
        unde_Area[k] = np.sum(X[i]['lbr'][X[i]['lbr'][:, k] < 0, k])                            # undershoot area

        # check if there are up and down slopes
        if peak_Posi[k] >= unde_Posi[k]:
            blacklist_idx.append(i)
        else:
            up_Slope[k] = np.max(np.diff(X[i]['lbr'][:peak_Posi[k], k]))
            down_Slope[k] = np.min(np.diff(X[i]['lbr'][peak_Posi[k]:unde_Posi[k], k]))

    X[i]['peak_Ampl'] = peak_Ampl
    X[i]['peak_Posi'] = peak_Posi
    X[i]['peak_Area'] = peak_Area
    X[i]['unde_Ampl'] = unde_Ampl
    X[i]['unde_Posi'] = unde_Posi
    X[i]['unde_Area'] = unde_Area
    X[i]['up_Slope'] = up_Slope
    X[i]['down_Slope'] = down_Slope

    # get NVC statistics for each simulation
    peak_NVC = np.zeros((K))
    for k in range(K):
        peak_NVC[k] = np.max(X[i]['cbf'][:, k])
    X[i]['peak_NVC'] = peak_NVC

    # get DCM statistics for each simulation
    peak_DCM = np.zeros((K))
    for k in range(K):
        peak_DCM[k] = np.max(X[i]['neuro'][:, k])
    X[i]['peak_DCM'] = peak_DCM

    blacklist = ['x_v', 'x_d']
    for key in blacklist:
        X[i].pop(key, None)

# remove simulations with NaN values
X = np.delete(X, blacklist_idx)
num_simulations = X.shape[0]

np.save(PATH + '/X.npy', X)

# plotting
LBR_all = []
for i in range(num_simulations):
    LBR_all.append(X[i]['lbr'].T)

# plt.figure()
# plt.suptitle('Laminar BOLD response')
# plt.subplot(2, 1, 1)
# plt.title('Mean')
# plt.imshow(np.mean(LBR_all, axis=0), aspect='auto', cmap='bwr')
# plt.colorbar()
# plt.subplot(2, 1, 2)
# plt.title('Standard Deviation')
# plt.imshow(np.std(LBR_all, axis=0), aspect='auto', cmap='bwr')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(PATH + '/LBR.png')

# plt.figure(figsize=(4, 8))
# for i in range(num_simulations):
#     plt.subplot(3, 1, 1)
#     plt.title('Superficial LBR')
#     plt.plot(X[i]['lbr'][:, 0], c='k', alpha=0.1)
#     plt.subplot(3, 1, 2)
#     plt.title('Middle LBR')
#     plt.plot(X[i]['lbr'][:, 1], c='k', alpha=0.1)
#     plt.subplot(3, 1, 3)
#     plt.title('Deep LBR')
#     plt.plot(X[i]['lbr'][:, 2], c='k', alpha=0.1)
# plt.tight_layout()
# plt.savefig(PATH + '/LBR_all.png')
