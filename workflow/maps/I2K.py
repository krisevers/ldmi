import numpy as np
import h5py
import json

def I2K(K, species, area, sigma=0):
    with open('maps/thickness.json', 'r') as f:
        DATA = json.load(f)

    # load recurrent synapse depth probabilities
    hf = h5py.File('maps/curr_syn_map.h5', 'r')
    PROB = hf['MAP'][:]
    hf.close()

    # get number of sources and targets
    num_sources = PROB.shape[2]
    num_targets = PROB.shape[1]

    human, macaque = 0, 1
    V1, V2, V3, V3A, MT = 0, 1, 2, 3, 4

    thickness = DATA['species'][macaque]['areas'][V1]['thickness']

    # get laminar boundaries
    norm_thickness = thickness / np.sum(thickness)

    # assign part of K to each layer
    K_per_layer = norm_thickness * K
    K_per_layer = np.round(K_per_layer).astype(int)

    # L1, L2/3, L4, L5, L6
    layers = [0, 1, 2, 3, 4]
    layer_to_K = np.repeat(layers, K_per_layer)

    PROB_K = np.zeros((K, PROB.shape[1], PROB.shape[2]))
    for k in range(K):
        PROB_K[k] = PROB[layer_to_K[k]]

    # remove depth if longer than K by interpolating
    for k in range(K):
        if PROB_K[k].shape[0] > K:
            PROB_K[k] = np.interp(np.linspace(0, 1, K), np.linspace(0, 1, PROB_K[k].shape[0]), PROB_K[k])


    if sigma != 0:
        from scipy.ndimage import gaussian_filter1d
        PROB_K_smooth = np.zeros_like(PROB_K)
        for t in range(num_targets):
            for s in range(num_sources):
                PROB_K_smooth[:, t, s] = gaussian_filter1d(PROB_K[:, t, s], sigma=sigma)
        PROB_K = PROB_K_smooth

    return PROB_K
    

if __name__=="__main__":

    import pylab as plt
    import IPython

    num_sources = 8 # L23E L23I L4E L4I L5E L5I L6E L6I
    num_targets = 8 # L23E L23I L4E L4I L5E L5I L6E L6I
    num_layers  = 5 # L1 L23 L4 L5 L6

    K = 13
    species = 'macaque'
    area = 'V1'

    PROB_K = I2K(K, species, area)


    # locations of population synapses
    layers = ['L1', 'L23', 'L4', 'L5', 'L6']
    populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    colors = plt.cm.Spectral(np.linspace(0, 1, num_sources))
    plt.figure(figsize=(12, 8))
    for t in range(num_targets):
        plt.subplot(4, 2, t+1)
        for s in range(num_sources):
            plt.plot(PROB_K[:, t, s], 'o-', color=colors[s], label=populations[s])
        plt.plot(np.nanmean(PROB_K[:, t, :], axis=1), 'o-', color='k')
        plt.title('Target {}'.format(populations[t]))
        plt.xlabel('Depth')
        plt.ylabel('Probability')
    plt.suptitle('Probability of synapse from source to target in depth')
    plt.tight_layout(pad=1)
    plt.savefig('maps/recurrent_synapse_depth_prob_K.png'.format(area))
    plt.show() 

    # smooth probability along cortical depth (K) with Gaussian kernel
    from scipy.ndimage import gaussian_filter1d
    PROB_K_smooth = np.zeros_like(PROB_K)
    for t in range(num_targets):
        for s in range(num_sources):
            PROB_K_smooth[:, t, s] = gaussian_filter1d(PROB_K[:, t, s], sigma=.7)

    # locations of population synapses
    layers = ['L1', 'L23', 'L4', 'L5', 'L6']
    populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    colors = plt.cm.Spectral(np.linspace(0, 1, num_sources))
    plt.figure(figsize=(12, 8))
    for t in range(num_targets):
        plt.subplot(4, 2, t+1)
        for s in range(num_sources):
            plt.plot(PROB_K_smooth[:, t, s], 'o-', color=colors[s], label=populations[s])
        plt.plot(np.nanmean(PROB_K_smooth[:, t, :], axis=1), 'o-', color='k')
        plt.xticks([0, K-1], ['CSF', 'WM'])
        plt.title('Target {}'.format(populations[t]))
        plt.xlabel('Depth')
        plt.ylabel('Probability')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2, title='Source')
    plt.suptitle('Probability of synapse from source to target in depth ({})'.format(area))
    plt.tight_layout(pad=1)
    plt.savefig('maps/recurrent_synapse_depth_prob_K_smooth_{}.png'.format(area))
    plt.show()


    IPython.embed()