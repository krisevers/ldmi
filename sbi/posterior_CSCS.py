import numpy as np
import pylab as plt

import torch
import sbi.utils as utils
from sbi.inference import SNPE, SNLE, SNRE
from sbi.analysis import pairplot, conditional_pairplot

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

    print('\n')
    print("##################################################################")
    print("                 LAMINAR DYNAMIC MODEL INFERENCE                  ")
    print("                 Training Posterior and Inference                 ")
    print("                       by Kris Evers, 2023                        ")
    print("##################################################################")


    import IPython

    import glob

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='Set path to obtain data')
    parser.add_argument('-m', '--models', nargs='+', default='DCM', help='Set models to explore parameters')

    args = parser.parse_args()

    PATH = 'data/' + args.path

    print('\n')
    print("##################################################################")
    print("Loading data...")

    theta_keys = ['U_1', 'U_2', 'U_3', 'dur', 
                  'c1', 'c2', 'c3', 
                  'w_v', 'V0_p', 't0v', 'tau_v_in', 'tau_v_de', 'tau_d_in', 'tau_d_de', 'tau_p_in', 'tau_p_de', 'Hct_v', 'Hct_d', 'Hct_p', 'R2s_t', 'R2s_v', 'R2s_d', 'R2s_p']

    data_keys = ['peak_Posi', 'peak_Ampl', 'peak_Area', 'unde_Posi', 'unde_Ampl', 'unde_Area', 'up_Slope', 'down_Slope']

    all_theta = []
    all_X = []
    for np_name in glob.glob(PATH + '/*.np[yz]'):
        X_ = np.load(np_name, allow_pickle=True)
        for i in range(X_.shape[0]):
            theta_i = []
            X_i = []
            for par in X_[i].keys():
                if par in theta_keys:
                    theta_i.append(X_[i][par])
                elif par in data_keys:
                    X_i.append(X_[i][par])
            all_theta.append(np.ravel(theta_i))
            all_X.append(np.ravel(X_i))

    print('\n')
    print("##################################################################")
    print("Preparing data...")

    # remove nans
    theta = []
    X = []
    for i in range(len(all_theta)):
        if np.isnan(all_theta[i]).any() or np.isnan(all_X[i]).any():
            pass
        else:
            theta.append(all_theta[i])
            X.append(all_X[i])

    num_simulations = len(X)

    theta = theta[:num_simulations]
    X = X[:num_simulations]

    theta = np.asarray(theta)
    X = np.asarray(X)
    X = X.reshape((num_simulations, -1))

    theta = torch.from_numpy(theta).float()
    X = torch.from_numpy(X).float()

    IPython.embed()

    # train
    print('\n')
    print("##################################################################")
    print("Training posterior...")
    posterior = train(num_simulations,
                        X,
                        theta,
                        num_threads=1,
                        method="SNPE",
                        device="cpu",
                        density_estimator="maf")

    # infer
    print('\n')
    print("##################################################################")
    print("Inferring posterior...")
    obs_x = X.mean(axis=0)
    num_samples = 10000

    posterior.set_default_x(obs_x)
    posterior_samples = posterior.sample((num_samples,))


    print('\n')
    print("##################################################################")
    print("Plotting results...")
    # pairplot
    fig, ax = pairplot(samples=posterior_samples, 
                       labels=theta_keys,
                       figsize=(20, 20)
    )
    plt.savefig(PATH + '/pairplot.png', dpi=300)

    # conditional pairplot
    mean_limit = theta.mean(axis=0).tolist()
    std_limit = theta.std(axis=0).tolist()
    limits = [(mean_limit[m] - 2*std_limit[m], mean_limit[m] + 2*std_limit[m]) for m in range(len(theta_keys))]

    condition = posterior.sample((1,))
    fig, ax = conditional_pairplot(
        density=posterior,
        condition=condition,
        limits=limits,
        labels=theta_keys,
        figsize=(20, 20),
    )
    plt.savefig(PATH + '/conditional_pairplot.png', dpi=300)

    # marginal correlation matrix
    corr_matrix_marginal = np.corrcoef(posterior_samples.T)
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    plt.title("Marginal Correlation Matrix")
    im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap="PiYG")
    _ = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(theta_keys)), theta_keys, rotation=45)
    plt.yticks(np.arange(len(theta_keys)), theta_keys)
    plt.tight_layout()
    plt.savefig(PATH + '/marginal_correlation_matrix.png', dpi=300)



    IPython.embed()