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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='data', help='Set path to obtain data')

    args = parser.parse_args()

    PATH = args.path

    # load data
    X = np.load(PATH + '/X.npy', allow_pickle=True)

    num_simulations = len(X)

    stats = np.empty((num_simulations, 15))
    theta = np.empty((num_simulations, 12))
    keys  = np.array(list(X[0]['theta'].keys()))

    for i in range(num_simulations):
        stats[i] = np.array(list(X[i]['stats'].values()))
        theta[i] = np.array(list(X[i]['theta'].values()))

    theta = np.asarray(theta)
    stats = np.asarray(stats)
    stats = stats.reshape((num_simulations, -1))

    # remove nans
    idx = np.argwhere(np.isnan(stats))
    stats = np.delete(stats, idx, axis=0)
    theta = np.delete(theta, idx, axis=0)
    print("Removed {} simulations with NaNs".format(len(idx)))
    
    theta = torch.from_numpy(theta).float()
    stats = torch.from_numpy(stats).float()

    print("Training posterior on {} simulations".format(num_simulations))

    posterior = train(num_simulations,
                    stats,
                    theta,
                    num_threads=1,
                    method="SNPE",
                    device="cpu",
                    density_estimator="maf")
    
    torch.save(posterior, PATH + '/PDCM_posterior.pt')

    import IPython; IPython.embed()