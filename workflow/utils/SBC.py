import numpy as np

def SBC(p_prior, p_sample, p_posterior, K, N, M, J):

    theta   = np.empty((N, M, K), dtype=np.float32)
    rank    = np.zeros((N, K))
    b       = np.zeros(J)

    for n in range(N):
        # simulate parameters and data
        p = p_prior(K)
        x = p_sample(p)

        # posterior draws given simulated data
        for m in range(M):
            p_post = p_posterior(x)
            theta[n, m, :] = p_post

        # calculate rank of sim among posterior draws
        for k in range(K):
            rank[n, k] = np.sum(theta[n, :, k] < x)


        # import IPython; IPython.embed();

        # calculate b
        for m in range(M):
            bin = (np.floor(rank[n, m] * J / (M+1))).astype(int)
            print(bin, rank[n, m])
            b[bin] += 1


    return rank, theta, b





if __name__=="__main__":

    import pylab as plt
    
    # define prior
    def p_prior(K):
        # K is the number of parameters
        return np.random.uniform(0, 1, size=K)

    # define data-generating distribution
    def p_sample(p):
        # p is a set of prior parameters on which the data-generating distribution depends
        return np.random.normal(p, 0.1, size=10)

    # define posterior
    def p_posterior(x):
        # function runs linear regression and returns posterior mean and variance
        return np.random.normal(x, 0.1, size=10)


    # run SBC
    rank, theta, b = SBC(p_prior, p_sample, p_posterior, K=10, N=1000, M=100, J=20)


    plt.figure()
    plt.imshow(rank, aspect='auto')
    plt.colorbar()
    plt.xlabel('posterior draw')
    plt.ylabel('simulation')
    plt.show()


    import IPython; IPython.embed();
