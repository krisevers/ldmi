import numpy as np
import pylab as plt

def pairplot(samples, labels=None, figsize=(10, 10), bounds=None):
    """
    Create a pairplot from samples.
    """
    num_samples, num_dims = samples.shape

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]

    fig, ax = plt.subplots(num_dims, num_dims, figsize=figsize)
    plt.suptitle(r'p($\theta | x$)', fontsize=20)
    for i in range(num_dims):
        for j in range(num_dims):
            if (i == j):
                ax[i, j].hist(np.array(samples[:, i]), bins=50, density=True, histtype="step", color="black")
                ax[i, j].set_xlabel(labels[j])
                ax[i, j].set_ylabel(labels[i])
                if bounds is not None:
                    ax[i, j].set_xlim(bounds[i])
                ax[i, j].set_yticks([])
            if (i < j):
                ax[i, j].hist2d(np.array(samples[:, j]), np.array(samples[:, i]), bins=50, cmap="Reds")
                ax[i, j].set_xlabel(labels[j])
                ax[i, j].set_ylabel(labels[i])
                if bounds is not None:
                    ax[i, j].set_xlim(bounds[j])
                    ax[i, j].set_ylim(bounds[i])
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            if (i > j):
                ax[i, j].axis("off")

    return fig, ax

def marginal(samples, labels=None, bounds=None, figsize=(8, 12)):
    """
    Create a marginal plot from samples.
    """
    num_samples, num_dims = samples.shape

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]

    fig, ax = plt.subplots(num_dims, 1, figsize=figsize)
    plt.suptitle(r'p($\theta | x$)', fontsize=20)
    for i in range(num_dims):
        if bounds is None:
            ax[i].hist(np.array(samples[:, i]), bins=50, density=True, histtype="step", color="black")
        else:   
            ax[i].hist(np.array(samples[:, i]), bins=50, density=True, histtype="step", color="black", range=bounds[i])
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel(r"$p(\theta_{} | x)$".format({i}))
        ax[i].set_yticks([])

    plt.tight_layout()

    return fig, ax

def marginal_correlation(samples, labels=None, figsize=(10, 10), bounds=None):
    """
    Create a marginal correlation matrix
    """

    num_samples, num_dims = samples.shape

    corr_matrix_marginal = np.corrcoef(samples.T)

    if (labels is None):
        labels = [r"$\theta_{}$".format(i) for i in range(num_dims)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.suptitle(r'p($\theta | x$)', fontsize=20)
    im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap="PiYG")
    ax.set_xticks(np.arange(num_dims))
    ax.set_yticks(np.arange(num_dims))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap="PiYG")
    _ = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax