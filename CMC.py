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
C *= 200

k = np.array([256, 128, 16, 32], dtype=np.float128)

# state variables (ss, sp, ii, dp)
v = np.zeros(4, dtype=np.float128)
r = np.zeros(4, dtype=np.float128)

dt = np.float128(0.001) # seconds
T  = np.float128(35)    # seconds

u = np.zeros((int(T/dt), 4))
u[int(5/dt):int(6/dt), 0] = np.float128(1)   

# save the state variables
V = np.zeros((int(T/dt), 4))
R = np.zeros((int(T/dt), 4))

def sigmoid(x, r=2/3):
    return 1 / (1 + np.exp(-r*x))

for t in range(int(T/dt)):
    # euler
    v_ = sigmoid(v)
    v_ = v_ - sigmoid(0)

    p = C * u[t,:]
    q = np.dot(G, v_)

    v = k * (r - 2*v - v)
    r = q + p

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

