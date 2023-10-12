import numpy as np

def get_L2K(K, L, area='V1'):

    # L1    L2/3 L4   L5   L6
    if area == 'V1':
        th = [0.08, 0.25, 0.37, 0.14, 0.16]
    elif area == 'MT':
        th = [0.11, 0.54, 0.13, 0.11, 0.11]
    
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
    
    L2K  = np.zeros((K,L))

    L2K[:int(bot_L23),            0] = 1
    L2K[int(top_L4):int(bot_L4),  1] = 1
    L2K[int(top_L5):int(bot_L5),  2] = 1
    L2K[int(top_L6):,             3] = 1

    return L2K, TH