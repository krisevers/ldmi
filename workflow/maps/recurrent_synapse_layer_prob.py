import numpy as np

# TODO: add thalamic (TH) and cortico-cortical (CC) external inputs

num_sources = 8 # L23E L23I L4E L4I L5E L5I L6E L6I
num_targets = 8 # L23E L23I L4E L4I L5E L5I L6E L6I
num_layers  = 5 # L1 L23 L4 L5 L6

MAP = np.zeros((num_layers, num_targets, num_sources))
# target: L23E
MAP[:, 0, :] = np.array([   [82.7,	3471.5,	0.1,	0,	    0],
                            [15.8,	755.6,	0,	    0,	    0],
                            [1.5,	878.9,	0.6,	0,	    0],
                            [0,	    44.7,	0.2,	0,	    0],
                            [0.9,	430.1,	0,	    0,	    0],
                            [0,	    0,	    0,	    0,	    0],
                            [0,	    135,	1.2,	0,	    0],
                            [0,	    0,	    0,	    0,	    0]]).T

# target: L23I
MAP[:, 1, :] = np.array([   [118,	3162.8,	0.3,	0,	    0],
                            [22.5,	957.9,	0.1,	0,	    0],
                            [2.2,	800.9,	2,	    0,	    0],
                            [0.1,	55.2,	0.6,	0,	    0],
                            [1.3,	391.9,	0.1,	0,	    0],
                            [0,	    0,	    0,	    0,	    0],
                            [0,	    123,    3.3,	0,	    0],
                            [0,	    0,	    0,	    0,	    0]]).T

# target: L4E
MAP[:, 2, :] = np.array([   [51,	695.8,	416.6,	86.8,	0],
                            [9.8,	101.9,	129.1,	2,	    0],
                            [0.9,	176.2,	3010,	24.2,	0],
                            [0,	    6.3,	1457.1,	1.1,	0],
                            [0.6,	86.2,	126.8,	24,	    0],
                            [0,	    0,	    0,	    2.6,	0],
                            [0,	    27,	    5021.9,	8.2,	0],
                            [0,	    0,	    0,	    0,	    0]]).T

# target: L4I
MAP[:, 3, :] = np.array([   [0,	    107.3,	81,  	0,	    0],
                            [0,	    15.7,	27.2,	0,	    0],
                            [0,	    36.2,	585.5,	0,	    0],
                            [0,	    1,	    348.1,	0,	    0],
                            [0,	    13.3,	24.7,	0,	    0],
                            [0,	    0,	    0,	    0,	    0],
                            [0,	    4.2,	976.8,	0,	    0],
                            [0,	    0,	    0,	    0,	    0]]).T

# target: L5E
MAP[:, 4, :] = np.array([   [370.1,	1122.1,	34.4,	4238.1,	5   ],
                            [70.6,	164.3,	9.5,	186.7,	0.3 ],
                            [6.8,	284.2,	248.7,	1186.3,	5.8 ],
                            [0.2,	10.2,	81.9,	82.5,	0.1 ],
                            [4.1,	139,	10.4,	1171.9,	10  ],
                            [0,	    0,	    0,	    174.9,	0.6 ],
                            [0,	    43.6,	414.9,	401.1,	29.1],
                            [0,	    0,	    0,	    0,	    0   ]]).T

# target: L5I
MAP[:, 5, :] = np.array([   [0,	    0,	    0,	    1357,	0],
                            [0,	    0,	    0,	    76.2,	0],
                            [0,	    0,	    0,	    379.8,	0],
                            [0,	    0,	    0,	    32.1,	0],
                            [0,	    0,	    0,	    375.2,	0],
                            [0,	    0,	    0,	    64.3,	0],
                            [0,	    0,	    0,	    128.5,	0],
                            [0,	    0,	    0,	    0,	    0]]).T

# target: L6E
MAP[:, 6, :] = np.array([   [3.9,	256.6,	60.3,	620,	218.4   ],
                            [0.8,	37.5,	16.5,	14,	    19.7    ],
                            [0.1,	64.9,	435.8,	173.5,	252.1   ],
                            [0,	    2.3,	143.5,	7.6,    6.9     ],
                            [0,	    31.8,	18.2,	171.4,	436.9   ],
                            [0,	    0,	    0,	    18.9,	29.1    ],
                            [0,	    10,	    727,	58.7,	1268.3  ],
                            [0,	    0,	    0,	    0,	    0       ]]).T

# target: L6I
MAP[:, 7, :] = np.array([   [0,	    0,	    0,	    0,	    79.6    ],
                            [0,	    0,	    0,	    0,	    7.7     ],
                            [0,	    0,	    0,	    0,	    91.9    ],
                            [0,	    0,	    0,	    0,	    2.7     ],
                            [0,	    0,	    0,	    0,	    159.3   ],
                            [0,	    0,	    0,	    0,	    11.3    ],
                            [0,	    0,	    0,	    0,	    462.1   ],
                            [0,	    0,	    0,	    0,	    0       ]]).T

if __name__ == '__main__':

    import pylab as plt
    
    # total number of synapses per connection
    tot_num_syn = np.sum(MAP, axis=0)

    # probability of synapse from source to target in layer
    PROB = np.zeros_like(MAP)
    for l in range(num_layers):
        PROB[l, :, :] = MAP[l, :, :] / np.nansum(MAP[l, :, :], axis=0) + 1e-9

    # replace nans with zeros
    PROB[np.isnan(PROB)] = 0

    np.save('maps/recurrent_synapse_layer_prob.npy', PROB)

    plt.figure()
    plt.imshow(tot_num_syn, interpolation='nearest', cmap='Reds')
    plt.colorbar()
    plt.title('Number of synapses per connection')
    plt.show()

    layers = ['L1', 'L23', 'L4', 'L5', 'L6']
    populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    plt.figure(figsize=(12, 6))
    for l in range(num_layers):
        plt.subplot(2, 3, l+1)
        plt.imshow(PROB[l, :, :], interpolation='nearest', cmap='Reds')
        plt.title('Layer {}'.format(layers[l]))
        plt.xticks(np.arange(num_sources), populations, rotation=90)
        plt.yticks(np.arange(num_targets), populations)
    plt.suptitle('Probability of synapse from source to target in layer')
    plt.tight_layout(pad=1)
    plt.show()


    # locations of population synapses
    layers = ['L1', 'L23', 'L4', 'L5', 'L6']
    populations = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    colors = plt.cm.Spectral(np.linspace(0, 1, num_sources))
    plt.figure(figsize=(12, 6))
    for t in range(num_targets):
        plt.subplot(4, 2, t+1)
        for s in range(num_sources):
            plt.plot(PROB[:, t, s], 'o-', color=colors[s], label=populations[s])
        plt.plot(np.nanmean(PROB[:, t, :], axis=1), 'o-', color='k')
        plt.title('Target {}'.format(populations[t]))
        plt.xticks(np.arange(num_layers), layers, rotation=90)
    plt.suptitle('Probability of synapse from source to target in layer')
    plt.tight_layout(pad=1)
    plt.savefig('recurrent_synapse_layer_prob.png')
    plt.show() 


    import IPython; IPython.embed()

