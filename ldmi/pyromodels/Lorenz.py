import argparse

import os
import sys
sys.path.insert(0, os.getcwd())

import pylab as plt

import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

def dz_dt(z, theta):
    """
    Lorenz system
    """
    x, y, z = z
    sigma, beta, rho = [theta[..., i] for i in range(3)]

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return jnp.array([dx_dt, dy_dt, dz_dt])

def model(T, y=None):
    """
    :param in T: number of time steps
    :param numpy.ndarray y: measured states with shape (T, 3)
    """
    # initial state
    z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([3]))
    # times
    ts = jnp.arange(T)
    # parameters sigma, beta, rho of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=jnp.array([0.0, 0.0, 0.0]),
            loc=jnp.array([10.0, 2.667, 28.0]),
            scale=jnp.array([1.0, 1.0, 1.0]),
        )
    )
    # integrate dz/dt, the result is of shape (T, 3)
    for t in range(1, T):
        z = numpyro.deterministic(f"z_{t}", z_init)
        z_init = z + dz_dt(z, theta) * 0.01
    # measurement error
    sigma_obs = numpyro.sample("sigma_obs", dist.LogNormal(jnp.log(0.1), 1).expand([3]))
    # measured states
    numpyro.sample("y", dist.Normal(z, sigma_obs), obs=y)

def main(args, data, dt):

    print("Running Lorenz model with NUTS kernel...")
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model, dense_mass=True),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(PRNGKey(1), T=data.shape[0], y=data)
    mcmc.print_summary()

    print("Predicting Lorenz model with NUTS kernel...")
    z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([3]))
    theta = mcmc.get_samples()["theta"]
    for t in range(data.shape[0], data.shape[0] + 10):
        z = numpyro.deterministic(f"z_{t}", z_init)
        z_init = z + dz_dt(z, theta) * dt

    # make plots
    plt.figure(figsize=(6, 6))
    plt.plot(mcmc.get_samples()["z_init"][:, 0], mcmc.get_samples()["z_init"][:, 2], "ro", ms=2, alpha=0.3)
    plt.plot(mcmc.get_samples()["z_init"][:, 0].mean(), mcmc.get_samples()["z_init"][:, 2].mean(), "ro")
    plt.plot(data[:, 0], data[:, 2], "kx", lw=0.5)
    plt.plot(data[:, 0].mean(), data[:, 2].mean(), "kx")
    plt.plot(mcmc.get_samples()["z_init"][:, 0].mean(), mcmc.get_samples()["z_init"][:, 2].mean(), "ro")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    # generate data
    import numpy as np
    from ldmi.models.Lorenz import Sim
    sigma, beta, rho = 10, 2.667, 28
    y0 = [0, 1, 1.05]
    t_sim = 10.
    dt = 1e-4
    L = Sim(dt=dt, t_sim=t_sim, y=y0, rho=rho, sigma=sigma, beta=beta)
    L.integrate('euler')
    T = L.get_times()
    X = L.get_states()
    data = np.asarray(X)
    print("simulation done! data shape: " + str(data.shape))
    

    parser = argparse.ArgumentParser(description="Lorenz attractor")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args, data, dt)