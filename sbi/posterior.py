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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='Set path to obtain data')
    parser.add_argument('-m', '--models', nargs='+', default='DCM', help='Set models to explore parameters')

    args = parser.parse_args()

    PATH = 'data/' + args.path

    X = np.load(PATH + '/X.npy', allow_pickle=True)


    params_of_interest = []
    if 'DCM' in args.models:
        DCM_paramlist = np.load(PATH + '/DCM_paramlist.npy', allow_pickle=True).tolist()
        params_of_interest += DCM_paramlist
    if 'NVC' in args.models:
        NVC_paramlist = np.load(PATH + '/NVC_paramlist.npy', allow_pickle=True).tolist()
        params_of_interest += NVC_paramlist
    if 'LBR' in args.models:
        LBR_paramlist = np.load(PATH + '/LBR_paramlist.npy', allow_pickle=True).tolist()
        params_of_interest += LBR_paramlist

    num_simulations = X.shape[0]

    sim_pars_keys  = ['K', 'T', 'dt']
    sum_stat_keys  = np.load(PATH + '/sum_stat_keys.npy', allow_pickle=True).tolist()
    data_keys      = ['neuro', 'cbf', 'lbr']

    print('\n')
    print("##################################################################")
    print("Number of simulations: {}".format(num_simulations))
    print("Summary statistics: {}".format(sum_stat_keys))
    print("Parameters: {}".format(params_of_interest))
    print("##################################################################")

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

    theta = np.asarray(theta)
    x = np.asarray(x)
    x = x.reshape((num_simulations, -1))

    theta = torch.from_numpy(theta).float()
    x = torch.from_numpy(x).float()

    # train
    print('\n')
    print("##################################################################")
    print("Training posterior...")
    posterior = train(num_simulations,
                        x,
                        theta,
                        num_threads=1,
                        method="SNPE",
                        device="cpu",
                        density_estimator="maf")

    # infer
    print('\n')
    print("##################################################################")
    print("Inferring posterior...")
    obs_x = x.mean(axis=0)
    num_samples = 10000

    posterior.set_default_x(obs_x)
    posterior_samples = posterior.sample((num_samples,))


    print('\n')
    print("##################################################################")
    print("Plotting results...")
    # pairplot
    fig, ax = pairplot(samples=posterior_samples, 
                       labels=params_of_interest,
                       figsize=(5, 5)
    )
    plt.savefig(PATH + '/pairplot.png')

    # conditional pairplot
    mean_limit = theta.mean(axis=0).tolist()
    std_limit = theta.std(axis=0).tolist()
    limits = [(mean_limit[m] - 2*std_limit[m], mean_limit[m] + 2*std_limit[m]) for m in range(len(params_of_interest))]

    condition = posterior.sample((1,))
    fig, ax = conditional_pairplot(
        density=posterior,
        condition=condition,
        limits=limits,
        labels=params_of_interest,
        figsize=(5, 5),
    )
    plt.savefig(PATH + '/conditional_pairplot.png')

    # marginal correlation matrix
    corr_matrix_marginal = np.corrcoef(posterior_samples.T)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.title("Marginal Correlation Matrix")
    im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap="PiYG")
    _ = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(params_of_interest)), params_of_interest, rotation=45)
    plt.yticks(np.arange(len(params_of_interest)), params_of_interest)
    plt.tight_layout()
    plt.savefig(PATH + '/marginal_correlation_matrix.png')



    IPython.embed()