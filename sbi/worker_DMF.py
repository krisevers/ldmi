import numpy as np

import simulators.DMF_model as DMF_model
import simulators.NVC_model as NVC_model
import simulators.LBR_model as LBR_model

from utils import get_N2K, gen_input


"""
Selection of parameter worker functions for DCM > NVC > LBR models
"""

def worker(K=12, T=20, DMF=None, test=False):
    """
    Wrapper function for DCM, NVC and LBR models
    """
    DMF_params = DMF
    # connectivity
    if 'P_L23E>L23E' in DMF_params.keys():
        DMF_params['P'][0, 0] *= DMF_params['P_L23E>L23E']
    if 'P_L4E>L23E' in DMF_params.keys():
        DMF_params['P'][0, 2] *= DMF_params['P_L4E>L23E']
    if 'P_L4E>L23I' in DMF_params.keys():
        DMF_params['P'][1, 2] *= DMF_params['P_L4E>L23I']
    if 'P_L4I>L23E' in DMF_params.keys():
        DMF_params['P'][0, 3] *= DMF_params['P_L4I>L23E']
    if 'P_L4I>L23I' in DMF_params.keys():
        DMF_params['P'][1, 3] *= DMF_params['P_L4I>L23I']
    if 'P_L23E>L4E' in DMF_params.keys():
        DMF_params['P'][2, 0] *= DMF_params['P_L23E>L4E']
    if 'P_L23E>L4I' in DMF_params.keys():
        DMF_params['P'][3, 0] *= DMF_params['P_L23E>L4I']
    if 'P_L23I>L4E' in DMF_params.keys():
        DMF_params['P'][2, 1] *= DMF_params['P_L23I>L4E']
    if 'P_L23I>L4I' in DMF_params.keys():
        DMF_params['P'][3, 1] *= DMF_params['P_L23I>L4I']
    if 'P_L23E>L5E' in DMF_params.keys():
        DMF_params['P'][4, 0] *= DMF_params['P_L23E>L5E']
    if 'P_L23E>L5I' in DMF_params.keys():
        DMF_params['P'][5, 0] *= DMF_params['P_L23E>L5I']
    if 'P_L5I>L23E' in DMF_params.keys():
        DMF_params['P'][1, 4] *= DMF_params['P_L5I>L23E']
    if 'P_L5I>L23I' in DMF_params.keys():
        DMF_params['P'][1, 5] *= DMF_params['P_L5I>L23I']
    if 'P_L6E>L4E' in DMF_params.keys():
        DMF_params['P'][2, 6] *= DMF_params['P_L6E>L4E']
    if 'P_L6E>L4I' in DMF_params.keys():
        DMF_params['P'][3, 6] *= DMF_params['P_L6E>L4I']

    DMF_params['K'] = np.log(1-DMF_params['P']) / np.log(1 - 1/(DMF_params['N'] * DMF_params['N'])) / DMF_params['N']
    DMF_params['W'] *= DMF_params['K']

    # external input
    U = {}
    onset = 3
    dur = 5
    amp = 1
    std = 1.5e-3
    U['u'] = np.zeros((int(T / DMF_params['dt']), DMF_params['M']))           # Matrix with input vectors to the neuronal model (one column per depth)
    if 'U_L23E' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 0, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L23E'], std=std)
    if 'U_L23I' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 1, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L23I'], std=std)
    if 'U_L4E' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 2, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L4E'], std=std)
    if 'U_L4I' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 3, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L4I'], std=std)
    if 'U_L5E' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 4, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L5E'], std=std)
    if 'U_L5I' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 5, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L5I'], std=std)
    if 'U_L6E' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 6, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L6E'], std=std)
    if 'U_L6I' in DMF_params.keys():
        U['u'] = gen_input(U['u'], 7, dt=DMF_params['dt'], start=onset, stop=onset+dur, amp=DMF_params['U_L6I'], std=std)

    # Buxton et al. (2004): The neural response is defined such that N(t) = 1 on the plateau of a sustained stimulus when no adaptation effects are operating. Similar to neuronal model defined in Havlicek et al. (2020).

    # default NVC parameters
    # NVC_params = {'K': K, 'dt': 1e-4, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6, 'T': T}
    NVC_params = {'K': K, 'dt': 1e-4, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6, 'T': T}

    # default LBR parameters
    LBR_params = {'K': K, 'dt': 0.01, 'T': T}
    LBR_params = LBR_model.LBR_parameters(K, LBR_params)
    LBR_params['alpha_v']  = 0.35
    LBR_params['alpha_d']  = 0.2
    LBR_params['tau_d_de'] = 30

    # simulate
    syn_signal = DMF_model.DMF_sim(U['u'], DMF_params)    # [T, M]

    # syn_signal[:, [0, 2, 4, 6]] = syn_signal[:, [0, 2, 4, 6]] + abs(syn_signal[:, [1, 3, 5, 7]])


    N2K, TH = get_N2K(K) # [K, M]

    neuro = np.zeros((syn_signal.shape[0], K))
    for t in range(np.shape(syn_signal)[0]):
        neuro[t] = np.sum(N2K*np.maximum(np.zeros(DMF_params['M']), syn_signal[t]), axis=1)    # switch sign of inhibitory synaptic activity

    # deviation from baseline (at t = onset-1/dt)
    neuro = neuro - neuro[onset - int(.1/DMF_params['dt'])]

    # remove simulation up to fixed point (t < 1/dt)
    neuro = neuro[int(1/DMF_params['dt']):]

    # multiply by N * surface area and divide by K
    # neuro = neuro * np.dot(N2K, DMF_params['N']) * .05   # voxel width = 0.3 mm
    # neuro /= TH

    neuro *= 100

    new_T = T - 1
    NVC_params['T'] = new_T
    LBR_params['T'] = new_T

    cbf = NVC_model.NVC_sim(neuro, NVC_params)

    from scipy import ndimage
    cbf = ndimage.gaussian_filter1d(cbf, K/(4*2), 1)

    new_dt = 0.01
    old_dt = DMF_params['dt']
    lbr_dt = new_dt/old_dt
    cbf_ = cbf[::int(lbr_dt)]

    LBR_params['dt'] = new_dt

    lbr, lbrpial, Y = LBR_model.LBR_sim(cbf_, LBR_params)

    lbr = lbr.astype(np.float64)
    lbrpial = lbrpial.astype(np.float64)

    # return {'neuro': neuro, 'cbf': cbf, 'lbr': lbr}

    if np.isnan(lbr).any() or np.isnan(cbf).any() or np.isnan(neuro).any():       # check for nan values
        peak_Posi = np.nan
        peak_Ampl = np.nan
        peak_Area = np.nan
        unde_Posi = np.nan
        unde_Ampl = np.nan
        unde_Area = np.nan
        up_Slope = np.nan
        down_Slope = np.nan
    else:   # compute summary features
        peak_Posi = np.zeros((K), dtype=int)
        peak_Ampl = np.zeros((K))
        peak_Area = np.zeros((K))
        unde_Posi = np.zeros((K), dtype=int)
        unde_Ampl = np.zeros((K))
        unde_Area = np.zeros((K))
        up_Slope = np.zeros((K))
        down_Slope = np.zeros((K))

        for k in range(K):
            peak_Ampl[k] = np.max(lbr[:, k])                                                # response peak amplitude
            peak_Posi[k] = int(np.where(lbr[:, k] == peak_Ampl[k])[0][0])                   # response peak latency
            peak_Area[k] = np.sum(lbr[lbr[:, k] > 0, k])                                    # response peak area
            unde_Ampl[k] = np.min(lbr[peak_Posi[k]:, k])                                    # undershoot amplitude
            unde_Posi[k] = int(np.where(lbr[:, k] == unde_Ampl[k])[0][0])                   # undershoot latency
            unde_Area[k] = np.sum(lbr[lbr[:, k] < 0, k])                                    # undershoot area

            # check if there are up and down slopes
            if peak_Posi[k] >= unde_Posi[k]:
                pass
            else:
                up_Slope[k] = np.max(np.diff(lbr[:peak_Posi[k], k]))
                down_Slope[k] = np.min(np.diff(lbr[peak_Posi[k]:unde_Posi[k], k]))

    if test == False:
        return {'peak_Posi':  peak_Posi, 
                'peak_Ampl':  peak_Ampl, 
                'peak_Area':  peak_Area, 
                'unde_Posi':  unde_Posi, 
                'unde_Ampl':  unde_Ampl, 
                'unde_Area':  unde_Area, 
                'up_Slope':   up_Slope, 
                'down_Slope': down_Slope,
                }
    
    else:
        return {'peak_Posi':  peak_Posi, 
                'peak_Ampl':  peak_Ampl, 
                'peak_Area':  peak_Area, 
                'unde_Posi':  unde_Posi, 
                'unde_Ampl':  unde_Ampl, 
                'unde_Area':  unde_Area, 
                'up_Slope':   up_Slope, 
                'down_Slope': down_Slope,
                'neuro': neuro,
                'cbf': cbf_,
                'lbr': lbr
                }

if __name__ == '__main__':

    PATH = 'data/'

    import mpi4py.MPI as MPI
    import tqdm as tqdm
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    values = []
    num_simulations = size*2

    num_simulations = 1


    K = 15          # number of cortical depths
    T_sim = 60      # simulation time

    # set DMF parameters
    DMF_blacklist = ['K', 'dt', 'T']
    DMF_paramlist = []
    DMF_theta = []
    for i in range(num_simulations):
        DMF_params = {'K': K, 'dt': 1e-4, 'T': T_sim}
        DMF_params = DMF_model.DMF_parameters(DMF_params)    # get default parameters
        # select parameters to explore
        # DMF_params['P_L23E>L23E'] = np.random.uniform(0, 2)
        # DMF_params['P_L4E>L23E']  = np.random.uniform(0, 2)
        # DMF_params['P_L4E>L23I']  = np.random.uniform(0, 2)
        # DMF_params['P_L4I>L23E']  = np.random.uniform(0, 2)
        # DMF_params['P_L4I>L23I']  = np.random.uniform(0, 2)
        # DMF_params['P_L23E>L4E']  = np.random.uniform(0, 2)
        # DMF_params['P_L23E>L4I']  = np.random.uniform(0, 2)
        # DMF_params['P_L23I>L4E']  = np.random.uniform(0, 2)
        # DMF_params['P_L23I>L4I']  = np.random.uniform(0, 2)
        # DMF_params['P_L23E>L5E']  = np.random.uniform(0, 2)
        # DMF_params['P_L23E>L5I']  = np.random.uniform(0, 2)
        # DMF_params['P_L5I>L23E']  = np.random.uniform(0, 2)
        # DMF_params['P_L5I>L23I']  = np.random.uniform(0, 2)
        # DMF_params['P_L6E>L4E']   = np.random.uniform(0, 2)
        # DMF_params['P_L6E>L4I']   = np.random.uniform(0, 2)
        # DMF_params['U_L23E']   = np.random.uniform(0,  15)
        # DMF_params['U_L23I']   = np.random.uniform(0,  15)
        # DMF_params['U_L4E']    = np.random.uniform(0,  15)
        # DMF_params['U_L4I']    = np.random.uniform(0,  15)
        # DMF_params['U_L5E']    = np.random.uniform(0,  15)
        # DMF_params['U_L5I']    = np.random.uniform(0,  15)
        # DMF_params['U_L6E']    = np.random.uniform(0,  15)
        # DMF_params['U_L6I']    = np.random.uniform(0,  15)
        DMF_params['U_L23E']      = np.random.uniform(0,  15)
        DMF_params['U_L23I']      = np.random.uniform(0,  15)
        DMF_params['U_L4E']       = np.random.uniform(0,  15)
        DMF_params['U_L4I']       = np.random.uniform(0,  15)
        DMF_params['U_L5E']       = np.random.uniform(0,  15)
        DMF_params['U_L5I']       = np.random.uniform(0,  15)
        DMF_params['U_L6E']       = np.random.uniform(0,  15)
        DMF_params['U_L6I']       = np.random.uniform(0,  15)
        DMF_theta.append(DMF_params)
    # save DMF exploration parameters 
    DMF_paramlist = ['P_L23E>L23E', 'P_L4E>L23E', 'P_L4E>L23I', 'P_L4I>L23E', 'P_L4I>L23I', 'P_L23E>L4E', 'P_L23E>L4I', 'P_L23I>L4E', 'P_L23I>L4I', 'P_L23E>L5E', 'P_L23E>L5I', 'P_L5I>L23E', 'P_L5I>L23I', 'P_L6E>L4E', 'P_L6E>L4I', 'U_L23E', 'U_L23I', 'U_L4E', 'U_L4I', 'U_L5E', 'U_L5I', 'U_L6E', 'U_L6I']


    NVC_theta = np.transpose([
        np.ones(num_simulations) * 0.6,  # c1 | default: 0.6
        np.ones(num_simulations) * 1.5,  # c2 | default: 1.5
        np.ones(num_simulations) * 0.6   # c3 | default: 0.6
    ])

    LBR_theta = np.transpose([
        np.full(num_simulations, None)
    ])

    DMF_paras_per_worker = np.array_split(DMF_theta, comm.Get_size())

    num_simulations_per_worker = int(num_simulations / size)
    DMF_workers_params = DMF_paras_per_worker[rank]

    X = []
    for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
        DMF = DMF_workers_params[i]
        X_i = worker(K=K, T=T_sim, DMF=DMF, test=True)
        X_i.update(DMF)
        X.append(X_i)

    if rank == 0:
        model_name = 'DMF_test'

        import pylab as plt
        import matplotlib.colors as colors
        import matplotlib.cbook as cbook
        from matplotlib import cm

        N2K, TH = get_N2K(K)
        TH_, ind = np.unique(TH, return_index=True)
        TH_ = TH_[np.argsort(ind)]

        for tr in range(len(X)):
            cm_neuro = plt.cm.Spectral(np.linspace(0, 1, K))
            cm_cbf   = plt.cm.Spectral(np.linspace(0, 1, K))
            cm_lbr   = plt.cm.Spectral(np.linspace(0, 1, K))
            plt.figure(figsize=(10, 10))
            plt.subplot(3, 1, 1)
            for i in range(K):
                plt.plot(X[tr]['neuro'][:,i], color=cm_neuro[i])
            plt.subplot(3, 1, 2)
            for i in range(K):
                plt.plot(X[tr]['cbf'][:, i], color=cm_cbf[i])
            plt.subplot(3, 1, 3)
            for i in range(K):
                plt.plot(X[tr]['lbr'][:, i], color=cm_lbr[i])
            plt.savefig('png/test/plot_neuro_cbf_lbr_ex{}.png'.format(tr))

            layers = ['L23', 'L4', 'L5', 'L6']
            layer_pos = [2, 7, 11, 14]
            plt.text(0, 2, layers[0], color='black', fontsize=20)
            plt.text(0, 7, layers[1], color='black', fontsize=20)
            plt.text(0, 11, layers[2], color='black', fontsize=20)
            plt.text(0, 14, layers[3], color='black', fontsize=20)

            plt.figure(figsize=(10, 10))
            plt.subplot(3, 1, 1)
            plt.imshow(X[tr]['neuro'].T, cmap='Spectral', aspect='auto', interpolation='none')
            plt.colorbar()
            for i in range(4):
                plt.text(1/1e-4, layer_pos[i], layers[i], color='black')
            plt.subplot(3, 1, 2)
            plt.imshow(X[tr]['cbf'].T, cmap='Spectral', aspect='auto', interpolation='none')
            plt.colorbar()
            for i in range(4):
                plt.text(1/1e-2, layer_pos[i], layers[i], color='black')
            plt.subplot(3, 1, 3)
            plt.imshow(X[tr]['lbr'].T, cmap='Spectral', aspect='auto', interpolation='none')
            plt.colorbar()
            for i in range(4):
                plt.text(1/1e-2, layer_pos[i], layers[i], color='black')
            plt.tight_layout()
            plt.savefig('png/test/imshow_neuro_cbf_lbr_ex{}.png'.format(tr))


        import IPython
        IPython.embed()
