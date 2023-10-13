import numpy as np

def get_L2K(K, L, area='V1'):

    # L1 L23 L4 L5 L6
    if area == 'V1':
        th = np.array([0.09, 0.37, 0.46, 0.17, 0.16])
    elif area == 'V2':
        th = np.array([0.12, 0.60, 0.24, 0.25, 0.25])
    elif area == 'VP':
        th = np.array([0.18, 0.63, 0.32, 0.21, 0.25])
    elif area == 'V3':
        th = np.array([0.23, 0.70, 0.31, 0.16, 0.19])
    elif area == 'V3A':
        th = np.array([0.20, 0.71, 0.24, 0.23, 0.28])
    elif area == 'MT':
        th = np.array([0.20, 0.95, 0.26, 0.26, 0.29])
    elif area == 'V4t':
        th = np.array([0.22, 0.80, 0.29, 0.26, 0.31])
    elif area == 'V4':
        th = np.array([0.18, 1.00, 0.24, 0.24, 0.24])
    elif area == 'VOT':
        th = np.array([0.23, 0.81, 0.28, 0.27, 0.32])
    elif area == 'MSTd':
        th = np.array([0.26, 0.92, 0.24, 0.30, 0.36])
    elif area == 'PIP':
        th = np.array([0.26, 0.92, 0.24, 0.30, 0.36])
    elif area == 'PO':
        th = np.array([0.26, 0.92, 0.24, 0.30, 0.36])
    elif area == 'DP':
        th = np.array([0.26, 0.91, 0.23, 0.30, 0.36])
    elif area == 'MIP':
        th = np.array([0.20, 0.85, 0.17, 0.16, 0.70])
    elif area == 'MDP':
        th = np.array([0.26, 0.92, 0.24, 0.30, 0.36])
    elif area == 'VIP':
        th = np.array([0.25, 1.17, 0.28, 0.21, 0.16])
    elif area == 'LIP':
        th = np.array([0.25, 1.00, 0.24, 0.24, 0.57])
    elif area == 'PITv':
        th = np.array([0.23, 0.81, 0.28, 0.27, 0.32])
    elif area == 'PITd':
        th = np.array([0.23, 0.81, 0.28, 0.27, 0.32])
    elif area == 'MSTl':
        th = np.array([0.26, 0.92, 0.24, 0.30, 0.36])
    elif area == 'CITv':
        th = np.array([0.29, 1.02, 0.19, 0.33, 0.40])
    elif area == 'CITd':
        th = np.array([0.29, 1.02, 0.19, 0.33, 0.40])
    elif area == 'FEF':
        th = np.array([0.22, 0.92, 0.35, 0.37, 0.35])
    elif area == 'TF':
        th = np.array([0.23, 0.66, 0.21, 0.24, 0.28])
    elif area == 'AITv':
        th = np.array([0.34, 1.20, 0.23, 0.39, 0.47])
    elif area == 'FST':
        th = np.array([0.51, 0.90, 0.18, 0.30, 0.36])
    elif area == '7a':
        th = np.array([0.35, 1.24, 0.21, 0.41, 0.48])
    elif area == 'STPp':
        th = np.array([0.29, 1.03, 0.18, 0.34, 0.40])
    elif area == 'STPa':
        th = np.array([0.29, 1.03, 0.18, 0.34, 0.40])
    elif area == '46':
        th = np.array([0.22, 0.82, 0.18, 0.28, 0.36])
    elif area == 'AITd':
        th = np.array([0.34, 1.20, 0.23, 0.39, 0.47])
    elif area == 'TH':
        th = np.array([0.28, 0.65, 0.12, 0.57, 0.26])
    
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

def get_N(area='V1'):

    if area == 'V1':
        N = np.array([47386, 13366, 70387, 17597, 20740, 4554, 19839, 4063]) / 2
    elif area == 'V2':
        N = np.array([50521, 14250, 36685, 9171, 19079, 4189, 19248, 3941]) / 2
    elif area == 'VP':
        N = np.array([52973, 14942, 49292, 12323, 15929, 3497, 19130, 3917]) / 2
    elif area == 'V3':
        N = np.array([58475, 16494, 47428, 11857, 12056, 2647, 14529, 2975]) / 2
    elif area == 'V3A':
        N = np.array([40887, 11532, 23789, 5947, 12671, 2782, 15218, 3116]) / 2
    elif area == 'MT':
        N = np.array([60606, 17095, 28202, 7050, 14176, 3113, 15837, 3243]) / 2
    elif area == 'V4t':
        N = np.array([48175, 13588, 34735, 8684, 14857, 3262, 17843, 3654]) / 2
    elif area == 'V4':
        N = np.array([64447, 18178, 33855, 8464, 13990, 3072, 14161, 2900]) / 2
    elif area == 'VOT':
        N = np.array([45313, 12781, 37611, 9403, 15828, 3475, 19008, 3892]) / 2
    elif area == 'MSTd':
        N = np.array([44343, 12507, 22524, 5631, 14742, 3237, 17704, 3625]) / 2
    elif area == 'PIP':
        N = np.array([44343, 12507, 22524, 5631, 14742, 3237, 17704, 3625]) / 2
    elif area == 'PO':
        N = np.array([44343, 12507, 22524, 5631, 14742, 3237, 17704, 3625]) / 2
    elif area == 'DP':
        N = np.array([43934, 12392, 18896, 4724, 14179, 3113, 17028, 3487]) / 2
    elif area == 'MIP':
        N = np.array([41274, 11642, 15875, 3969, 7681, 1686, 34601, 7086]) / 2
    elif area == 'MDP':
        N = np.array([44343, 12507, 22524, 5631, 14742, 3237, 17704, 3625]) / 2
    elif area == 'VIP':
        N = np.array([56683, 15988, 26275, 6569, 10099, 2217, 7864, 1610]) / 2
    elif area == 'LIP':
        N = np.array([51983, 14662, 20095, 5024, 11630, 2554, 28115, 5757]) / 2
    elif area == 'PITv':
        N = np.array([45313, 12781, 37611, 9403, 15828, 3475, 19008, 3892]) / 2
    elif area == 'PITd':
        N = np.array([45313, 12781, 37611, 9403, 15828, 3475, 19008, 3892]) / 2
    elif area == 'MSTl':
        N = np.array([44343, 12507, 22524, 5631, 14742, 3237, 17704, 3625]) / 2
    elif area == 'CITv':
        N = np.array([41696, 11761, 15303, 3826, 14385, 3158, 17275, 3537]) / 2
    elif area == 'CITd':
        N = np.array([41696, 11761, 15303, 3826, 14385, 3158, 17275, 3537]) / 2
    elif area == 'FEF':
        N = np.array([44053, 12425, 23143, 5786, 16943, 3720, 16128, 3302]) / 2
    elif area == 'TF':
        N = np.array([30774, 8680, 17143, 4286, 11082, 2433, 13310, 2725]) / 2
    elif area == 'AITv':
        N = np.array([49224, 13884, 18066, 4516, 16982, 3729, 20395, 4176]) / 2
    elif area == 'FST':
        N = np.array([36337, 10249, 12503, 3126, 12624, 2772, 15160, 3104]) / 2
    elif area == '7a':
        N = np.array([49481, 13957, 13279, 3320, 15817, 3473, 18996, 3890]) / 2
    elif area == 'STPp':
        N = np.array([41677, 11755, 13092, 3273, 14218, 3122, 17075, 3496]) / 2
    elif area == 'STPa':
        N = np.array([41677, 11755, 13092, 3273, 14218, 3122, 17075, 3496]) / 2
    elif area == '46':
        N = np.array([32581, 9190, 10645, 2661, 11850, 2602, 15841, 3244]) / 2
    elif area == 'AITd':
        N = np.array([49224, 13884, 18066, 4516, 16982, 3729, 20395, 4176]) / 2

    return N