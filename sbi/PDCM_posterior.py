import numpy as np
import pylab as plt

import torch
import sbi.utils as utils
from sbi.inference import SNPE, SNLE, SNRE
# from sbi.analysis import pairplot, conditional_pairplot


def train(num_simulations,
          x, 
          theta,
          num_threads=1,
          method="SNPE",
          device="cpu",
          density_estimator="maf"):
    
    torch.set_num_threads(num_threads)

    if (len(x.shape) == 1):
        x = x[:, None]
    if (len(theta.shape) == 1):
        theta = theta[:, None]


    if (method == "SNPE"):
        inference = SNPE(
            density_estimator=density_estimator, device=device
        )
    elif (method == "SNLE"):
        inference = SNLE(
            density_estimator=density_estimator, device=device
        )
    elif (method == "SNRE"):
        inference = SNRE(
            density_estimator=density_estimator, device=device
        )
    else:
        raise ValueError("Unknown inference method")
    
    inference = inference.append_simulations(theta, x)
    _density_estimator = inference.train()
    posterior = inference.build_posterior(_density_estimator)

    return posterior

def infer(obs_stats,
          num_samples,
          posterior):
    return posterior.sample((num_samples,), x=obs_stats)

if __name__=="__main__":

    import IPython

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='data/PDCM', help='Set path to obtain data')

    args = parser.parse_args()

    PATH = args.path

    # load data
    X = np.load(PATH + '/X.npy', allow_pickle=True)

    num_simulations = len(X)

    stats = np.empty((num_simulations, 6))
    theta = np.empty((num_simulations, 12))
    keys  = np.array(list(X[0]['theta'].keys()))

    for i in range(num_simulations):
        stats[i] = X[i]['stats']
        theta[i] = np.array(list(X[i]['theta'].values()))

    theta = np.asarray(theta)
    stats = np.asarray(stats)
    stats = stats.reshape((num_simulations, -1))

    # remove nans
    idx = np.argwhere(np.isnan(stats))
    stats = np.delete(stats, idx, axis=0)
    theta = np.delete(theta, idx, axis=0)

    theta = torch.from_numpy(theta).float()
    stats = torch.from_numpy(stats).float()

    posterior = train(num_simulations,
                    stats,
                    theta,
                    num_threads=1,
                    method="SNPE",
                    device="cpu",
                    density_estimator="maf")

    obs_theta = np.array([0.6, 1.5, 0.6, 2, 4, 0.32, 0.4, 4, 0.0463, 0.191, 126.3, 0.028])
    obs_x = np.array([ 7.147,       5.40525194, 18.776,      -0.28197114,  1.327,      -0.0698291 ])
    num_samples = 10000

    posterior.set_default_x(obs_x)
    posterior_samples = posterior.sample((num_samples,))



    from view import pairplot, marginal_correlation, marginal

    keys = np.array([r'$c_{1}$', r'$c_{2}$', r'$c_{3}$', r'$\tau_{mtt}$', r'$\tau_{vs}$', 
                     r'$\alpha$', r'$E_0$', r'$V_0$', r'$\epsilon$', r'$\rho_0$', 
                     r'$\nu_0$', r'$TE$'])
    
    limits = np.array([
        [0.3,       0.9   ],
        [1,         2     ],
        [0.3,       0.9   ],
        [1,         5     ],
        [0.1,      30     ],
        [0.1,       0.5   ],
        [0.1,       0.8   ],
        [1,        10     ],
        [0.3390,    0.3967],
        [10,     1000     ],
        [40,      440     ],
        [0.015,     0.040 ],
    ])


    fig, ax = pairplot(posterior_samples, labels=keys)
    plt.savefig('svg/pairplot.svg', dpi=300)

    fig, ax = marginal_correlation(posterior_samples, labels=keys, figsize=(10, 10))
    plt.savefig('svg/marginal_correlation.svg', dpi=300)

    # NVC
    fig, ax = pairplot(posterior_samples[:, :3], labels=keys[:3])
    plt.savefig('svg/pairplot_NVC.svg', dpi=300)
    # BOLD 
    fig, ax = pairplot(posterior_samples[:, 3:], labels=keys[3:])
    plt.savefig('svg/pairplot_BOLD.svg', dpi=300)

    fig, ax = marginal(posterior_samples, labels=keys, limits=limits, figsize=(8, 12))
    for i in range(len(keys)):
        ax[i].axvline(obs_theta[i], color='r', linestyle='--')
    plt.savefig('svg/marginal.svg', dpi=300)

    IPython.embed()
