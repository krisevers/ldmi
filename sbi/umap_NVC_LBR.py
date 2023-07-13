import numpy as np
import pylab as plt

import IPython

PATH = 'data/'

X = np.load(PATH + 'X_NVC_LBR.npy', allow_pickle=True)

num_simulations = X.shape[0]

sim_pars_keys  = ['K', 'T', 'dt']
sum_stat_keys  = ['peak_Ampl', 'peak_Posi', 'peak_Area', 'unde_Ampl', 'unde_Posi', 'unde_Area', 'peak_NVC', 'peak_DCM']
data_keys      = ['cbf', 'lbr'] # neuro same for all simulations
params_of_interest = ['c1', 'c2', 'c3', 'alpha_v', 'alpha_d', 'tau_d_de']

theta = []
x = []
for i in range(num_simulations):
    theta_i = []
    x_i = []
    for par in X[i].keys():
        if par in sim_pars_keys:
            continue
        elif par in sum_stat_keys:
            x_i.append(X[i][par])
        elif par in params_of_interest:
            theta_i.append(X[i][par])
    theta.append(theta_i)
    x.append(x_i)

import umap.umap_ as umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', verbose=True)

x = np.asarray(x)
theta = np.asarray(theta)
x = x.reshape((num_simulations, -1))

umap_results = reducer.fit_transform(x)

# plotting
plt.figure()
plt.scatter(umap_results[:, 0], umap_results[:, 1], c=theta[:, 0], s=5, cmap='Spectral')
plt.colorbar()
plt.show()

IPython.embed()