import cmdstanpy
cmdstanpy.install_cmdstan()
import matplotlib.pyplot as plt
import numpy as np

# Load the Stan model
model = cmdstanpy.CmdStanModel(stan_file='lorenz.stan')

# Parameters for the Lorenz system
sigma = 10
rho = 28
beta = 8/3
theta = [sigma, rho, beta]

# Initial condition
y0 = [1, 1, 1]  # Starting close to the origin

# Time settings
T = 100  # Total time
N = 1000  # Number of points to simulate

# Running the simulation
fit = model.sample(data={'T': N, 'y0': y0, 'theta': theta}, 
                   fixed_param=True, 
                   iter_sampling=1, 
                   iter_warmup=0, 
                   chains=1)

# Extracting the simulated data
y = fit.stan_variable('y')

# Plotting the trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(y[:, 0], y[:, 1], y[:, 2])
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()