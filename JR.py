import numpy as np

"""
Implementation of Jansen-Rit model of the neocortex. (Jansen & Rit, 1998)
"""

y = np.zeros(6) # synaptic activity

A = 3.25 # max amplitude of excitatory PSP
B = 22 # max amplitude of inhibitory PSP

a = 100 # excitatory time constant
b = 50  # inhibitory time constant

C = 135 # global coupling strength
C1 = C
C2 = C*0.8
C3 = C*0.25
C4 = C*0.25

def sigmoid(x, r=0.56, v0=6, e0=2.5):
    return 2*e0 / (1 + np.exp(-r*(v0-x)))

dt = 0.001 # seconds
T = int(10/dt)

Y = np.zeros((int(10/dt), 6))

p = np.zeros(T)
p[int(0/dt):int(10/dt)] = np.random.uniform(120, 320, int(10/dt))

for t in range(T):
    # euler
    y[0] += dt * y[3]
    y[1] += dt * y[4]
    y[2] += dt * y[5]

    y[3] += dt * (A*a*sigmoid(y[1] - y[2]) - 2*a*y[3] - a**2*y[0])
    y[4] += dt * (A*a*(p[t] + C2*sigmoid(C1*y[0])) - 2*a*y[4] - a**2*y[1])
    y[5] += dt * (B*b*C4*sigmoid(C3*y[0]) - 2*b*y[5] - b**2*y[2])

    Y[t,:] = y

import pylab as plt
plt.figure()
plt.plot(Y[:,0], label='y0')
plt.plot(Y[:,1], label='y1')
plt.plot(Y[:,2], label='y2')
plt.plot(Y[:,3], label='y3')
plt.plot(Y[:,4], label='y4')
plt.plot(Y[:,5], label='y5')
plt.xlim(100, T)
plt.legend()

plt.show()
