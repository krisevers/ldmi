import numpy as np
import h5py

import torch
import sbi.utils as utils
from sbi.inference import SNPE, SNLE, SNRE

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
    parser.add_argument('-p', '--path',    default='data',  help='Set data path')
    parser.add_argument('-n', '--name',                     help='Name of experiment')
    parser.add_argument('-t', '--threads', default=1,       help='Number of threads to use for training', type=int)
    parser.add_argument('-m', '--method',  default='SNPE',  help='Inference method')
    parser.add_argument('-d', '--device',  default='cpu',   help='Device to use for training')
    parser.add_argument('-s', '--seed',    default=0,       help='Random seed')
    parser.add_argument('-r', '--ratio',   default=0.8,     help='Ratio of training and test set')

    args = parser.parse_args()

    PATH = args.path + '/' + args.name + '/'

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading data from {}...".format(PATH))
    hf = h5py.File(PATH + 'data.h5', 'r')
    PSI     = np.array(hf.get('PSI'))
    THETA   = np.array(hf.get('THETA'))
    bounds  = np.array(hf.get('bounds'))
    keys = np.array(hf.get('keys'))

    PSI = np.array(hf.get('MAP'))

    hf.close()

    num_simulations = np.shape(THETA)[0]

    THETA_torch = torch.from_numpy(THETA).float()
    PSI_torch   = torch.from_numpy(PSI).float()

    # split data into training and test set
    idx = np.random.permutation(num_simulations)
    idx_train = idx[:int(args.ratio * num_simulations)]
    idx_test  = idx[int(args.ratio * num_simulations):]

    THETA_train = THETA_torch[idx_train]
    PSI_train   = PSI_torch[idx_train]
    THETA_test  = THETA_torch[idx_test]
    PSI_test    = PSI_torch[idx_test]

    posterior = train(num_simulations,
                        PSI_train,
                        THETA_train,
                        num_threads         = args.threads,
                        method              = args.method,
                        device              = args.device,
                        density_estimator   = "maf")
    
    torch.save(posterior, PATH + 'posterior.pt')

    hf = h5py.File(PATH + 'data.h5', 'a')
    hf.create_dataset('THETA_train',    data=THETA_train)
    hf.create_dataset('PSI_train',      data=PSI_train)
    hf.create_dataset('THETA_test',     data=THETA_test)
    hf.create_dataset('PSI_test',       data=PSI_test)
    hf.close()