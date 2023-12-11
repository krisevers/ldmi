import numpy as np
import pylab as plt

from numba import jit

@jit(nopython=True)
def f(u, c, Delta_u, u_th):
    return c * np.exp((u - u_th) / Delta_u)

@jit(nopython=True)
def Pfire(u, c, Delta_u, u_th, lam_old, dt):
    lam = f(u, c, Delta_u, u_th)
    p_lam = .5 * (lam_old + lam) * dt

    if p_lam > 0.01:
        p_lam = -np.expm1(-p_lam)

    return p_lam, lam

def RDM(T, dt, dt_rec, params, I_ext, Nrecord, seed):

    rng1 = np.random.RandomState(seed)

    M   = params["M"]
    N   = params["N"]

    mu      = params["mu"]
    Delta_u = params["Delta_u"]
    c       = params["c"]
    vreset  = params["vreset"]
    vth     = params["vth"]
    tref    = params["tref"]
    delay   = params["delay"]
    n_ref   = (tref/dt).astype(int)
    n_delay = (delay/dt).astype(int)
    
    # membrane time constants
    tau  = params['tau_m']
    dtau = dt / tau

    # synaptic time constants
    Etaus = np.zeros(M)
    for m in range(M):
        if params["tau_s"][m] > 0:
            Etaus[m] = np.exp(-dt / params["tau_s"][m])

    weights = params["weights"]

    I = np.zeros((M, int(T/dt)))
    for i in range(M):
        I[i, int(I_ext["onset"][i]/dt):int(I_ext["offset"][i]/dt)] = I_ext["I"][i]

    # quantities to be recorded
    Nsteps = int(T/dt)
    Nsteps_rec = int(T/dt_rec)
    Nbin = int(dt_rec/dt)
    Abar = np.zeros((Nrecord, Nsteps_rec))
    A = np.zeros((Nrecord, Nsteps_rec))

    # initialization
    L = np.zeros(M, dtype=int)
    for i in range(M):
        L[i] = int(np.round((5 * tau[i] + tref[i]) / dt)) + 1
    Lmax = np.max(L)
    S = np.ones((M, Lmax))
    u = np.tile(vreset, (Lmax, 1)).T
    n = np.zeros((M, Lmax))
    lam = np.zeros((M, Lmax))
    x = np.zeros(M)
    y = np.zeros(M)
    z = np.zeros(M)
    for i in range(M):
        n[i, L[i]-1] = 1.

    h = vreset * np.ones(M)
    lambdafree = np.zeros(M)
    for i in range(M):
        lambdafree[i] = f(h[i], c[i], Delta_u[i], vth[i])

    # begin main simulation loop
    for ti in range(Nsteps):
        if ti % (Nsteps/100) == 1:
            print(int(100*ti/Nsteps), end="\r")

        t = dt*ti
        i_rec = int(ti/Nbin)

        synInput = np.zeros(M)
        for i in range(M):
            for j in range(M):
                synInput[i] += weights[j,i] * y[j] + I[i,ti]

        for i in range(M):
            h[i] += dtau[i]*(mu[i]-h[i]) + synInput[i] * dt
            Plam, lambdafree[i] = Pfire(h[i], c[i], Delta_u[i], vth[i], lambdafree[i], dt)
            W = Plam * x[i]
            X = x[i]
            Z = z[i]
            Y = Plam * z[i]
            z[i] = (1-Plam)**2 * z[i] + W
            x[i] -= W

            for l in range(1, L[i]-n_ref[i]):
                u[i, l-1] = u[i,l] + dtau[i] * (mu[i] - u[i,l])  + synInput[i] * dt
                Plam, lam[i,l-1] = Pfire(u[i,l-1], c[i], Delta_u[i], vth[i], lam[i,l], dt)
                m = S[i, l] * n[i,l]
                v = (1 - S[i,l]) * m
                W += Plam * m
                X += m
                Y += Plam * v
                Z += v
                S[i,l-1] = (1 - Plam) * S[i, l]
                n[i,l-1] = n[i,l]
            x[i] += S[i,0] * n[i, 0]
            z[i] += (1 - S[i,0]) *  S[i,0] * n[i, 0]
            for l in range(L[i]-n_ref[i], L[i]):
                X+=n[i,l]
                n[i,l-1]=n[i,l]

            if Z>0:
                PLAM = Y/Z
            else:
                PLAM = 0

            nmean = max(0, W +PLAM * (1 - X))
            if nmean>1:
                nmean = 1

            distrib = rng1.binomial(N[i], nmean)
            n[i, L[i]-1] = distrib / N[i] # population activity (fraction of neurons spiking)

            y[i] = y[i] * Etaus[i] + n[i, L[i]-1 - n_delay[i]] / dt * (1 - Etaus[i])

            if i < Nrecord:
                Abar[i, i_rec] += nmean
                A[i,i_rec] += n[i,L[i]-1]
    Abar /= (Nbin * dt)
    A  /= (Nbin * dt)

    return Abar, A


if __name__ == '__main__':

    params = {
        "M": 2,
        "N":            np.array([1000, 200]),
        "mu":           np.array([20.0, 20.0]),
        "Delta_u":      np.array([1.0, 1.0]),
        "c":            np.array([10.0, 10.0]),
        "vreset":       np.array([0.0, 0.0]),
        "vth":          np.array([10.0, 10.0]),
        "tref":         np.array([0.0, 0.0]),
        "delay":        np.array([0.0, 0.0]),
        "tau_m":        np.array([0.02, 0.02]),
        "tau_s":        np.array([0.0, 0.0]),
        "weights":      np.array([[0.3, -0.4], 
                                  [0.3, -0.4]]),
    }

    I_ext = {"onset":   np.array([0.5, 0.0]),
             "offset":  np.array([0.7, 0.0]),
             "I":       np.array([200.0, 0.0])}

    T = 1.0
    dt = 5e-4
    dt_rec = 1e-3
    Nrecord = 2
    seed = 0
    
    Abar, A = RDM(T, dt, dt_rec, params, I_ext, Nrecord, seed)

    plt.figure()
    plt.plot(Abar[0,:], 'b', label='Excitatory')
    plt.plot(Abar[1,:], 'r', label='Inhibitory')
    plt.show()
