import numpy as np
import pylab as plt

import IPython

"""
Implementation of the canonical microcircuit (CMC) model of the neocortex.
"""

G = np.array([[-8, -4, -4,  0],
              [ 4, -8, -2,  0],
              [ 4,  2, -4, -2],
              [ 0,  1, -2, -4]], dtype=np.float128)
G *= 200

C = np.array([ 1,  1,  1,  1], dtype=np.float128)

k   = np.array([256, 128, 16, 32], dtype=np.float128)
tau = np.array([2, 2, 16, 28], dtype=np.float128)
tau /= 1000

# state variables (ss, sp, ii, dp)
v = np.zeros(4, dtype=np.float128)  # voltage
r = np.zeros(4, dtype=np.float128)  # conductance

dt = np.float128(0.001) # seconds
T  = np.float128(35)    # seconds

u = np.zeros((int(T/dt), 4))
u[int(5/dt):int(6/dt), 0] = np.float128(1)   

# save the state variables
V = np.zeros((int(T/dt), 4))
R = np.zeros((int(T/dt), 4))

def sigmoid(x, r=1):
    return 1 / (1 + np.exp(-r*x))


for t in range(int(T/dt)):
    # euler
    F = 1 / (1 + np.exp(-1 * v + 0))
    S = F - 1/(1 + np.exp(0))

    U = np.dot(G, S) + C * u[t,:]
    f = (U - 2*r - v / tau) / tau
    
    V[t,:] = v
    R[t,:] = r


# plot the results
plt.figure()
plt.plot(V[:,0], label='ss')
plt.plot(V[:,1], label='sp')
plt.plot(V[:,2], label='ii')
plt.plot(V[:,3], label='dp')
plt.legend()

plt.figure()
plt.plot(R[:,0], label='ss')
plt.plot(R[:,1], label='sp')
plt.plot(R[:,2], label='ii')
plt.plot(R[:,3], label='dp')
plt.legend()

plt.show()


# other implementation of CMC

# tau = np.array([2, 2, 16, 18])

# G = np.array([[-800, -800, -1600,    0],
#               [ 800, -800,  -800,    0],
#               [ 800,  400,  -800,  400],
#               [   0,    0,  -400, -400]], dtype=np.float128)

# vdd = np.array([0, 0, 0, 0], dtype=np.float128)
# vd  = np.array([0, 0, 0, 0], dtype=np.float128)
