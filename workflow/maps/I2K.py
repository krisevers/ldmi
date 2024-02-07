import numpy as np
import h5py
import json

def get_thickness(K, species='macaque', area='V1'):
    with open('maps/thickness.json', 'r') as f:
        THICK = json.load(f)

    thickness = THICK[species][area]

    # normalize thickness
    norm_thickness = thickness / np.sum(thickness)
    # assign part of K to each layer
    K_per_layer = norm_thickness * K
    K_per_layer = np.ceil(K_per_layer).astype(int)

    # L1, L2/3, L4, L5, L6
    layers = ['1', '23', '4', '5', '6']
    layer_to_K = np.repeat(layers, K_per_layer)
    layer_to_K = layer_to_K[:K]

    return layer_to_K


def I2K(K, species='macaque', area='V1', sigma=0):

    with open('maps/thickness.json', 'r') as f:
        THICK = json.load(f)

    thickness = THICK[species][area]

    K_up = K * 3    # upsample K to get smooth probability
    K_up = K

    # normalize thickness
    norm_thickness = thickness / np.sum(thickness)
    # assign part of K to each layer
    K_per_layer = norm_thickness * K_up
    K_per_layer = np.ceil(K_per_layer).astype(int)

    # L1, L2/3, L4, L5, L6
    layers = ['1', '23', '4', '5', '6']
    layer_to_K = np.repeat(layers, K_per_layer)
    layer_to_K = layer_to_K[:K_up]

    # load recurrent synapse to layer probabilities
    with open('maps/synmap.json', 'r') as f:
        SYN = json.load(f)

    POPS    = SYN['pops']      # cell-type to population mapping
    SOURCES = SYN['sources']   # sources in model
    TARGETS = SYN['targets']   # targets in model
    DATA    = SYN['data']      # synaptic probability data
    LAYERS  = SYN['layers']    # layers in data

    # get probability of finding a synapse at a particular depth for each source-target pair and the number of synapses per neuron
    num_layers = len(LAYERS)
    PROB = np.zeros((K_up, len(TARGETS), len(SOURCES)))
    NUMSYN = np.zeros((num_layers, len(TARGETS)))
    OCCURRENCE = np.zeros(len(TARGETS))
    for p, pop in POPS.items():
        for y, target in enumerate(TARGETS):                # post-synaptic population
            if target in pop:
                OCCURRENCE[y] = DATA[target]['occurrence']
                for l, layer in enumerate(LAYERS):
                    if layer in DATA[target]['syn_dict'].keys():
                        NUMSYN[l, y] = DATA[target]['syn_dict'][layer]['number of synapses per neuron']
                        for x, source in enumerate(SOURCES):
                            PROB[layer_to_K == layer, y, x] = DATA[target]['syn_dict'][layer][source] / 100


    # get number of synapses per layer per target neuron per connection
    NUMSYN_K = np.zeros((K_up, len(TARGETS), len(SOURCES)))
    for l, layer in enumerate(LAYERS):
        for t, target in enumerate(TARGETS):
            NUMSYN_K[layer_to_K == layer, t, :] = NUMSYN[l, t] * PROB[layer_to_K == layer, t, :]

    # pool similar celltypes according to model populations
    model_sources = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I', 'TH', 'CC']
    model_targets = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    NUMSYN_KK = np.zeros((K_up, len(model_targets), len(model_sources)))
    for t, target in enumerate(model_targets):
        for s, source in enumerate(model_sources):
            pop_sources = POPS[source]
            pop_targets = POPS[target]
            for pop_source in pop_sources:
                for pop_target in pop_targets:
                    NUMSYN_KK[:, t, s] += NUMSYN_K[:, TARGETS.index(pop_target), SOURCES.index(pop_source)]

    # get probability of finding a synapse at a certain depth for each source-target pair
    PROB_KK = np.zeros((K_up, len(model_targets), len(model_sources)))
    for t, target in enumerate(model_targets):
        for s, source in enumerate(model_sources):
            if np.sum(NUMSYN_KK[:, t, s]) == 0:
                PROB_KK[:, t, s] = np.zeros(K_up)
            else:
                PROB_KK[:, t, s] = NUMSYN_KK[:, t, s] / np.sum(NUMSYN_KK[:, t, s])

    # smooth along cortical depth (K) with Gaussian kernel
    if sigma != 0:
        from scipy.ndimage import gaussian_filter1d
        PROB_KK_smooth = np.zeros_like(PROB_KK)
        for t, target in enumerate(model_targets):
            for s, source in enumerate(model_sources):
                PROB_KK_smooth[:, t, s] = gaussian_filter1d(PROB_KK[:, t, s], sigma=sigma)
                # ensure sum to 1
                if np.sum(PROB_KK_smooth[:, t, s]) == 0:
                    PROB_KK_smooth[:, t, s] = np.zeros(K_up)
                else:
                    PROB_KK_smooth[:, t, s] /= np.sum(PROB_KK_smooth[:, t, s])
        PROB_KK = PROB_KK_smooth

    # downsample from K_up to K
    PROB_K = np.zeros((K, len(model_targets), len(model_sources)))
    for t in range(len(model_targets)):
        for s in range(len(model_sources)):
            PROB_K[:, t, s] = np.mean(PROB_KK.reshape((K, K_up//K, len(model_targets), len(model_sources))), axis=1)[:, t, s]

    # probabilities for each source-target pair should sum to 1
    for t, target in enumerate(model_targets):
        for s, source in enumerate(model_sources):
            if np.sum(PROB_K[:, t, s]) == 0:
                PROB_K[:, t, s] = np.zeros(K)
            else:
                PROB_K[:, t, s] /= np.sum(PROB_K[:, t, s])

    with open('maps/popsize.json', 'r') as f:
        POPSIZE = json.load(f)

    # get population sizes
    popsize = POPSIZE[area]

    tot_popsize = np.sum(popsize)
    rel_popsize = popsize / tot_popsize

    rel_thickness = thickness / np.sum(thickness)

    for i in range(len(model_targets)):
        PROB_K[:, i, :] *= 0.25 #rel_popsize[i]

    return PROB_K


if __name__=="__main__":

    import pylab as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import IPython

    num_sources = 10 # L23E L23I L4E L4I L5E L5I L6E L6I TH CC
    num_targets = 8  # L23E L23I L4E L4I L5E L5I L6E L6I
    num_layers  = 5  # L1 L23 L4 L5 L6

    K = 31
    species = 'macaque'
    area = 'V1'

    PROB_K = I2K(K, species, area, sigma=10)

    num_targets = PROB_K.shape[1]
    num_sources = PROB_K.shape[2]

    targets = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    sources = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I', 'TH', 'CC']


    plt.figure(figsize=(6, 10))
    for t in range(num_targets):
        plt.subplot(4, 2, t+1)
        plt.title('Target: {}'.format(targets[t]))
        plt.xlabel('Source')
        plt.ylabel('Depth')
        im = plt.imshow(PROB_K[:, t], aspect='auto', cmap='Reds')
        plt.axvline(x=1.5, color='k', linestyle='--')           # L23 border
        plt.axvline(x=3.5, color='k', linestyle='--')           # L4 border
        plt.axvline(x=5.5, color='k', linestyle='--')           # L5 border
        plt.axvline(x=7.5, color='k', linestyle='-')            # external input
        plt.xticks(range(num_sources), sources, rotation=90)
        plt.yticks([0, K-1], ['CSF', 'WM'])
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig('pdf/I2K.pdf', dpi=300)
    plt.show()

    plt.figure(figsize=(3, 7))
    plt.subplot(4, 1, 1)
    plt.plot(np.sum(PROB_K[:, 0], axis=1), np.arange(K), color='b', label='L23E', lw=3)
    plt.plot(np.sum(PROB_K[:, 1], axis=1), np.arange(K), color='r', label='L23I', lw=3)
    plt.gca().invert_yaxis()
    plt.subplot(4, 1, 2)
    plt.plot(np.sum(PROB_K[:, 2], axis=1), np.arange(K), color='b', label='L4E', lw=3)
    plt.plot(np.sum(PROB_K[:, 3], axis=1), np.arange(K), color='r', label='L4I', lw=3)
    plt.gca().invert_yaxis()
    plt.subplot(4, 1, 3)
    plt.plot(np.sum(PROB_K[:, 4], axis=1), np.arange(K), color='b', label='L5E', lw=3)
    plt.plot(np.sum(PROB_K[:, 5], axis=1), np.arange(K), color='r', label='L5I', lw=3)
    plt.gca().invert_yaxis()
    plt.subplot(4, 1, 4)
    plt.plot(np.sum(PROB_K[:, 6], axis=1), np.arange(K), color='b', label='L6E', lw=3)
    plt.plot(np.sum(PROB_K[:, 7], axis=1), np.arange(K), color='r', label='L6I', lw=3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('pdf/I2K_sum.pdf', dpi=300)
    plt.show()



    # save to file
    with h5py.File('maps/I2K_{}_{}_K{}.h5'.format(species, area, K), 'w') as hf:
        hf.create_dataset('PROB', data=PROB_K)

    hf.close()