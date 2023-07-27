import numpy as np
from scipy.stats import norm


def get_N2K(K, area='V1'):

    # L1    L2/3 L4   L5   L6
    if area == 'V1':
        th = [0.08, 0.25, 0.37, 0.14, 0.16]
    elif area == 'MT':
        th = [0.11, 0.54, 0.13, 0.11, 0.11]

    N = 8
    
    # L2/3, L4, L5, L6
    th_L1  = th[0]
    th_L23 = th[1]
    th_L4  = th[2]
    th_L5  = th[3]
    th_L6  = th[4]

    th_all = th_L1 + th_L23 + th_L4 + th_L5 + th_L6

    rt_L1  = th_L1/th_all
    rt_L23 = th_L23/th_all
    rt_L4  = th_L4/th_all
    rt_L5  = th_L5/th_all
    rt_L6  = th_L6/th_all

    cntr_L23 = K*(rt_L1 + rt_L23/2)
    std_L23  = K*rt_L23/2
    top_L23  = 0
    bot_L23  = K*(rt_L1 + rt_L23)
    siz_L23  = int(bot_L23) - int(top_L23)

    cntr_L4  = K*(rt_L1 + rt_L23 + rt_L4/2)
    std_L4   = K*rt_L4/2
    top_L4   = bot_L23
    bot_L4   = top_L4 + K*(rt_L4)
    siz_L4   = int(bot_L4) - int(top_L4)

    cntr_L5  = K*(rt_L1 + rt_L23 + rt_L4 + rt_L5/2)
    std_L5   = K*rt_L5/2
    top_L5   = bot_L4
    bot_L5   = top_L5 + K*(rt_L5)
    siz_L5   = int(bot_L5) - int(top_L5)

    cntr_L6  = K*(rt_L1 + rt_L23 + rt_L4 + rt_L5 + rt_L6/2)
    std_L6   = K*rt_L6/2
    top_L6   = bot_L5
    bot_L6   = top_L6 + K*(rt_L6)
    siz_L6   = int(bot_L6) - int(top_L6)

    TH = np.concatenate([siz_L23*np.ones(siz_L23), siz_L4*np.ones(siz_L4), siz_L5*np.ones(siz_L5), siz_L6*np.ones(siz_L6)])    
    
    N2K  = np.zeros([K,N])
  
    # N2K[:,0] = norm(cntr_L23,std_L23).pdf(np.arange(K))
    # N2K[:,2] = norm(cntr_L4, std_L4).pdf(np.arange(K))
    # N2K[:,4] = norm(cntr_L5, std_L5).pdf(np.arange(K))
    # N2K[:,6] = norm(cntr_L6, std_L6).pdf(np.arange(K))

    N2K[int(top_L23):int(bot_L23),[0, 1]] = 1
    N2K[int(top_L4):int(bot_L4),[2, 3]]   = 1
    N2K[int(top_L5):int(bot_L5),[4, 5]]   = 1
    N2K[int(top_L6):int(bot_L6),[6, 7]]   = 1

    return N2K, TH

def gen_input(U, target, dt, start, stop, amp, std):

    intervidx_U = [int(start/dt), int(stop/dt)]
    peakstd_U = int(std/dt)
    INP = norm(peakstd_U*3, peakstd_U).pdf(np.arange(peakstd_U*6))
    INP = (INP - min(INP)) / (max(INP) - min(INP))
    U[intervidx_U[0]:intervidx_U[1], target] = max(INP)
    U[intervidx_U[0]:intervidx_U[0]+peakstd_U*3-1, target] = INP[1:peakstd_U*3]
    U[intervidx_U[1]:intervidx_U[1]+peakstd_U*3-1, target] = INP[peakstd_U*3+1:]
    U[:, target] = U[:, target]*amp
    return U

if __name__=="__main__":

    import pylab as plt

    K = 30

    N2K = get_N2K(K)

    plt.figure()
    plt.imshow(N2K, aspect='auto', cmap='hot')
    plt.savefig('png/N2K_K30.png')
    plt.close('all')