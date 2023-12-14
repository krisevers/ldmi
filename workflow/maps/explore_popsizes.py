import numpy as np
import json

with open('maps/popsize.json') as f:
    popsizes = json.load(f)

keys = list(popsizes.keys())

L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I = [], [], [], [], [], [], [], []

pops = [L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I]

for key in keys:
    for i, pop in enumerate(pops):
        pop.append(popsizes[key][i])



import pylab as plt 

colors = plt.cm.Spectral(np.linspace(0, 1, len(pops)))
labels = ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']

vmin = np.amin(pops)
vmax = np.amax(pops)

bins = np.linspace(vmin, vmax, 100)

max_per_pop = np.zeros(len(pops))
min_per_pop = np.zeros(len(pops))
for i, pop in enumerate(pops):
    max_per_pop[i] = np.max(pop)
    min_per_pop[i] = np.min(pop)

import IPython; IPython.embed()

plt.figure()
plt.plot(np.arange(len(pops)), max_per_pop, 'o', label='max')
plt.plot(np.arange(len(pops)), min_per_pop, 'o', label='min')
plt.legend()
plt.xticks(np.arange(len(pops)), labels)
plt.title('Maximum population size')
plt.show()

plt.figure()
plt.title('Population sizes')
for i, pop in enumerate(pops):
    mu, std = np.mean(pop), np.std(pop)
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    y = len(pop) * std * np.sqrt(2*np.pi)**-1 * np.exp(-(x - mu)**2 / (2*std**2))
    # normalize
    y /= np.sum(y)
    plt.plot(x, y, label=labels[i], color=colors[i])
    plt.xlim(0, vmax)
    plt.ylim(0, np.max(y) * 1.1)
plt.legend()
plt.show()