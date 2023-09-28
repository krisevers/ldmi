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

    # Excitatory contribution
    N2K[int(top_L23):int(bot_L23),0] = 1
    N2K[int(top_L4):int(bot_L4),  2] = 1
    N2K[int(top_L5):int(bot_L5),  4] = 1
    N2K[int(top_L6):int(bot_L6),  6] = 1

    # Inhibitory contribution
    N2K[int(top_L23):int(bot_L23),1] = 1
    N2K[int(top_L4):int(bot_L4),  3] = 1
    N2K[int(top_L5):int(bot_L5),  5] = 1
    N2K[int(top_L6):int(bot_L6),  7] = 1

    return N2K, TH

def gen_input(U, target, dt, start, stop, amp, std):

    intervidx_U = [int(start/dt), int(stop/dt)]
    peakstd_U = int(std/dt)
    INP = norm(peakstd_U*3, peakstd_U).pdf(np.arange(peakstd_U*6))
    INP = (INP - min(INP)) / (max(INP) - min(INP))
    if hasattr(target, '__len__'):
        for i in range(len(target)):
            U[intervidx_U[0]:intervidx_U[1], target[i]] = max(INP)
            U[intervidx_U[0]:intervidx_U[0]+peakstd_U*3-1, target[i]] = INP[1:peakstd_U*3]
            U[intervidx_U[1]:intervidx_U[1]+peakstd_U*3-1, target[i]] = INP[peakstd_U*3+1:]
            U[intervidx_U[0]:intervidx_U[1]+peakstd_U*3-1, target[i]] = U[intervidx_U[0]:intervidx_U[1]+peakstd_U*3-1, target[i]]*amp
    else:
        U[intervidx_U[0]:intervidx_U[1], target] = max(INP)
        U[intervidx_U[0]:intervidx_U[0]+peakstd_U*3-1, target] = INP[1:peakstd_U*3]
        U[intervidx_U[1]:intervidx_U[1]+peakstd_U*3-1, target] = INP[peakstd_U*3+1:]
        U[:, target] = U[:, target]*amp
    return U

def syn_to_neuro(syn, K, E_scale=1, I_scale=1, baseline=0):

    N2K, TH = get_N2K(K)

    N2K[:, ::2]  *= E_scale
    N2K[:, 1::2] *= I_scale
    
    neuro = np.zeros((int(np.shape(syn)[0]), K))
    for t in range(np.shape(neuro)[0]):
        neuro[t] = np.sum(N2K*abs(syn[t]), axis=1)

    # substract baseline
    neuro = neuro - neuro[baseline]
    
    return neuro

def gen_observables(lbr, E, O):
    
    num_volumes = np.shape(lbr)[0]
    num_layers  = np.shape(lbr)[1]

    num_stimulations = len(E['stimulations'])

    if np.isnan(lbr).any(): # check for nan values
        peak_Posi  = np.nan
        peak_Ampl  = np.nan
        peak_Area  = np.nan
        unde_Posi  = np.nan
        unde_Ampl  = np.nan
        unde_Area  = np.nan

    else:
        # per stimulation and layer generate features
        peak_Posi  = np.zeros((num_stimulations,  E['K']), dtype=int)
        peak_Ampl  = np.zeros((num_stimulations,  E['K']))
        peak_Area  = np.zeros((num_stimulations,  E['K']))
        unde_Posi  = np.zeros((num_stimulations,  E['K']), dtype=int)
        unde_Ampl  = np.zeros((num_stimulations,  E['K']))
        unde_Area  = np.zeros((num_stimulations,  E['K']))

        for s in range(num_stimulations):
            for k in range(num_layers):
                onset_idx = int(np.floor(E['stimulations'][s]['onset']/E['TR']))           # onset time in volume index
                dur_idx   = int(np.floor(E['stimulations'][s]['duration']/E['TR']))        # duration in number of volumes
                aft_idx   = 2*E['TR']                                                      # after stimulus period in number of volumes
                end_idx   = onset_idx + dur_idx + aft_idx                                  # end time in volume index

                peak_Ampl[s, k] = np.max(lbr[onset_idx:end_idx, k])                                     # response peak amplitude
                peak_Posi[s, k] = int(np.where(lbr[onset_idx:end_idx, k] == peak_Ampl[k])[0][0])        # response peak latency
                peak_Area[s, k] = np.sum(lbr[lbr[onset_idx:end_idx, k] > 0, k])                         # response peak area
                unde_Ampl[s, k] = np.min(lbr[peak_Posi[k]:end_idx, k])                                  # undershoot amplitude
                unde_Posi[s, k] = int(np.where(lbr[onset_idx:end_idx, k] == unde_Ampl[k])[0][0])        # undershoot latency
                unde_Area[s, k] = np.sum(lbr[lbr[onset_idx:end_idx, k] < 0, k])                         # undershoot area

    return {'peak_Posi':  peak_Posi, 
            'peak_Ampl':  peak_Ampl, 
            'peak_Area':  peak_Area, 
            'unde_Posi':  unde_Posi, 
            'unde_Ampl':  unde_Ampl, 
            'unde_Area':  unde_Area, 
            }

def create_theta(num_simulations, components=['DCM', 'NVC', 'LBR'], parameters=[[],[],[]], path=None, info=False):
    """
    Loads the prior bounds from json file corresponding to the components.
    Sets default value if parameter is not in tunable_parameters argument. 
    Creates sets of initialized parameters for all simulations to be run (theta).

    Parameters
    ----------
    num_simulations : int
        Number of simulations to be run.
    components : list, optional
        List of components to be used. The default is ['DCM', 'NVC', 'LBR'].
    theta_parameters : list, optional
        List of parameters to be tuned. The default is [[],[],[]].
    path : str, optional
        Path to the prior bounds json files. The default is None.

    Returns
    -------
    theta : dict
        Dictionary of initialized parameters for all simulations to be run.
    """

    import json
    import os

    if path is None:
        path = os.path.dirname(os.path.abspath(__file__)) + '/priors/'

    priors = {}
    theta = []

    # load priors
    for component in components:
        with open(path + component + '_priors.json', 'r') as f:
            priors[component] = json.load(f)

    # create theta
    for n in range(num_simulations):    # create theta for all simulations
        theta.append({})
        for component in components:
            theta[-1][component] = {}

            for parameter in priors[component].keys():
                if parameter in parameters[components.index(component)]:
                    theta[-1][component][parameter] = np.random.uniform(priors[component][parameter]['bounds']['lower'], 
                                                                        priors[component][parameter]['bounds']['upper'])

    return theta

if __name__=="__main__":

    import pylab as plt

    K = 30

    N2K = get_N2K(K)

    plt.figure()
    plt.imshow(N2K, aspect='auto', cmap='hot')
    plt.savefig('png/N2K_K30.png')
    plt.close('all')
