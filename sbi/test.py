import torch
from torch.distributions import MultivariateNormal

from sbi.analysis import ActiveSubspace
from sbi.simulators import linear_gaussian
from sbi.inference import simulate_for_sbi, infer

from utils import pairplot, marginal_correlation
import numpy as np
import pylab as plt

_ = torch.manual_seed(0)

prior = MultivariateNormal(0.0 * torch.ones(2), 2 * torch.eye(2))


def simulator(theta):
    return linear_gaussian(
        theta, -0.8 * torch.ones(2), torch.tensor([[1.0, 0.98], [0.98, 1.0]])
    )


theta_obs = torch.zeros(2)

posterior = infer(simulator, prior, num_simulations=2000, method="SNPE").set_default_x(
    theta_obs
)

posterior_samples = posterior.sample((20000,))

fig, ax = pairplot(posterior_samples, figsize=(4, 4))
plt.show()

sensitivity = ActiveSubspace(posterior)
e_vals, e_vecs = sensitivity.find_directions(posterior_log_prob_as_property=True)

print("Eigenvalues: \n", e_vals, "\n")
print("Eigenvectors: \n", e_vecs)

projected_data = sensitivity.project(posterior_samples, num_dimensions=1)

# project eigenvectors in 2D space using quiver
origin = np.zeros(2)
plt.figure()
plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], s=1, alpha=0.5)
plt.quiver(
    origin,
    origin,
    e_vecs[0, 0]*1,
    e_vecs[1, 0]*1,
    color="r",
    scale=3,
    label="1st eigenvector",
)
plt.quiver(
    origin,
    origin,
    e_vecs[0, 1]*1,
    e_vecs[1, 1]*1,
    color="b",
    scale=3,
    label="2nd eigenvector",
)
plt.legend()
plt.show()

import IPython; IPython.embed()