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

    if device == "cuda":
        # check if GPU is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Warning: CUDA GPU not available. Using CPU instead.")
            device = "cpu"
    
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

    # ratio to float
    ratio = float(args.ratio)

    print("Loading data from {}...".format(PATH))
    hf = h5py.File(PATH + 'data.h5', 'r')
    BETA    = np.array(hf.get('BETA'))
    THETA   = np.array(hf.get('THETA'))
    hf.close()

    print("Number of simulations: {}".format(np.shape(THETA)[0]))
    num_simulations = np.shape(THETA)[0]

    THETA_torch = torch.from_numpy(THETA).float()
    BETA_torch   = torch.from_numpy(BETA).float()

    print("Splitting data into training and test set...")
    if (ratio > 1.0 or ratio < 0.0):
        raise ValueError("Ratio must be between 0 and 1")
    elif (num_simulations < ratio*100):
        raise ValueError("Not enough simulations to split into training and test set")
    elif (num_simulations < 10000):
        print("Warning: Number of simulations is small. Consider using more simulations.")
    idx = np.random.permutation(num_simulations)
    idx_train = idx[:int(ratio * num_simulations)]
    idx_test  = idx[int(ratio * num_simulations):]

    THETA_train  = THETA_torch[idx_train]
    BETA_train   = BETA_torch[idx_train]
    THETA_test   = THETA_torch[idx_test]
    BETA_test    = BETA_torch[idx_test]

    print("Training {} on {} simulations...".format(args.method, num_simulations))
    posterior = train(num_simulations,
                        BETA_train,
                        THETA_train,
                        num_threads         = args.threads,
                        method              = args.method,
                        device              = args.device,
                        density_estimator   = "maf",
                        )
    

    print('Saving data...')
    torch.save(posterior, PATH + 'posterior.pt')
    hf = h5py.File(PATH + 'data.h5', 'a')
    hf.create_dataset('THETA_train',    data=THETA_train)
    hf.create_dataset('BETA_train',      data=BETA_train)
    hf.create_dataset('THETA_test',     data=THETA_test)
    hf.create_dataset('BETA_test',       data=BETA_test)
    hf.close()
