import numpy as np

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
    parser.add_argument('-p', '--path',    default='data/', help='Set path to obtain data')
    parser.add_argument('-n', '--name',    help='Name of experiment')
    parser.add_argument('-t', '--threads', default=1, help='Number of threads to use for training')
    parser.add_argument('-m', '--method',  default='SNPE', help='Inference method')
    parser.add_argument('-d', '--device',  default='cpu', help='Device to use for training')
    parser.add_argument('-s', '--seed',    default=0, help='Random seed')


    # ratio training and test set
    parser.add_argument('-r', '--ratio', default=0.8, help='Ratio of training and test set')


    args = parser.parse_args()

    PATH = args.path

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    X = np.load(PATH + args.name + '.npy', allow_pickle=True)

    num_simulations = len(X)

    keys  = np.array(list(X[0]['theta'].keys()))
    keys = ['I_L23E', 'I_L23I', 'I_L4E', 'I_L4I', 'I_L5E', 'I_L5I', 'I_L6E', 'I_L6I']
    num_theta = len(keys)
    num_psi   = len(np.array(np.concatenate(list(X[0]['Psi'].values()))))
    psi   = np.empty((num_simulations, num_psi))
    theta = np.empty((num_simulations, num_theta))


    for i in range(num_simulations):
        psi[i]   = np.array(np.concatenate(list(X[i]['Psi'].values())))
        for j in range(num_theta):
            theta[i, j] = X[i]['theta'][keys[j]]

    theta = np.asarray(theta)
    psi   = np.asarray(psi)
    psi   = psi.reshape((num_simulations, -1))

    # remove nans
    idx   = np.argwhere(np.isnan(psi))
    psi   = np.delete(psi, idx, axis=0)
    theta = np.delete(theta, idx, axis=0)
    print("Removed {} simulations with NaNs".format(len(idx)))

    # remove infs
    idx   = np.argwhere(np.isinf(psi))
    psi   = np.delete(psi, idx, axis=0)
    theta = np.delete(theta, idx, axis=0)
    print("Removed {} simulations with Infs".format(len(idx)))
    
    theta = torch.from_numpy(theta).float()
    psi   = torch.from_numpy(psi).float()

    num_simulations = len(theta)        # update number of simulations

    # split into training and test set
    idx = np.random.permutation(num_simulations)
    idx_train = idx[:int(args.ratio * num_simulations)]
    idx_test  = idx[int(args.ratio * num_simulations):]
    theta_train = theta[idx_train]
    psi_train   = psi[idx_train]

    theta_test = theta[idx_test]
    psi_test   = psi[idx_test]

    print("Training posterior on {} simulations".format(num_simulations))

    posterior = train(num_simulations,
                    psi_train,
                    theta_train,
                    num_threads = args.threads,
                    method      = args.method,
                    device      = args.device,
                    density_estimator="maf")
    
    torch.save(posterior, PATH + args.name + '_posterior.pt')
    np.save(PATH + args.name + '_psi_train.npy',    psi_train.numpy())
    np.save(PATH + args.name + '_theta_train.npy',  theta_train.numpy())
    np.save(PATH + args.name + '_psi_test.npy',     psi_test.numpy())
    np.save(PATH + args.name + '_theta_test.npy',   theta_test.numpy())

    import IPython; IPython.embed()