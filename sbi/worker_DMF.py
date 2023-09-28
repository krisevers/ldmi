import numpy as np

from simulators.DMF import DMF
from simulators.NVC import NVC
from simulators.LBR import LBR

from utils import get_N2K, gen_input, syn_to_neuro, gen_observables, create_theta


"""
Selection of parameter worker functions for DCM > NVC > LBR models
"""

def worker(E, O, theta=None, test=False):
    """
    Wrapper function for DCM, NVC and LBR models

    E - experiment parameters (dict)
    O - observables (dict)
    theta - parameters (dict)
    """

    # # load default models
    DMF_model = DMF()
    NVC_model = NVC()
    LBR_model = LBR(E)

    # set theta parameters
    if theta is not None:
        for component in theta.keys():
            if component == 'DMF':
                for parameter in theta[component].keys():
                    DMF_model.P[parameter] = theta[component][parameter]
            elif component == 'NVC':
                for parameter in theta[component].keys():
                    NVC_model.P[parameter] = theta[component][parameter]
            elif component == 'LBR':
                for parameter in theta[component].keys():
                    LBR_model.P[parameter] = theta[component][parameter]

    # external input
    E['U'] = np.zeros((int(E['T'] / DMF_model.P['dt']), DMF_model.P['M']))           # Matrix with input vectors to the neuronal model (one column per depth)
    for i in range(len(E['stimulations'])):
        onset  = E['stimulations'][i]['onset']
        dur    = E['stimulations'][i]['duration']
        amp    = E['stimulations'][i]['amplitude']
        std    = 1.5e-3
        target = E['stimulations'][i]['target']
        E['U'] = gen_input(E['U'], target, dt=DMF_model.P['dt'], start=onset, stop=onset+dur, amp=amp, std=std)

    # simulate
    syn_signal = DMF_model.sim(E)    # [T, M]

    syn_signal = np.maximum(syn_signal, 0)  # remove negative values

    # convert synaptic activity to neural response
    idx_baseline = int(1/DMF_model.P['dt'])  # ms
    neuro = syn_to_neuro(syn_signal, E['K'], E_scale=10, I_scale=0, baseline=idx_baseline)  # [T, K]

    # remove simulation up to fixed point (t < 1/dt)
    neuro = neuro[int(1/DMF_model.P['dt']):]
    E['T'] -= 1

    cbf = NVC_model.sim(neuro, E)

    # smooth cbf signal across depth (using gaussian kernel)
    from scipy.ndimage import gaussian_filter1d
    cbf = gaussian_filter1d(cbf, sigma=1, axis=1)

    # subsample cbf to match LBR dt
    lbr_dt = LBR_model.P['dt']
    old_dt = DMF_model.P['dt']
    new_dt = lbr_dt/old_dt
    cbf_   = cbf[::int(new_dt)]

    lbr, lbrpial, Y = LBR_model.sim(cbf_, E)

    lbr = lbr.astype(np.float64)
    lbrpial = lbrpial.astype(np.float64)

    # downsample lbr to TR
    lbr_ = lbr[::int(E['TR']/lbr_dt)]

    # TODO: downsample K depths to 3 voxels (i.e. to match fMRI resolution) by interpolation
    
    if test:
        X = {'syn_signal': syn_signal, 'neuro': neuro, 'cbf': cbf_, 'lbr': lbr_, 'lbrpial': lbrpial, 'Y': Y}
        X_obs = gen_observables(lbr_, E, O={})
        X.update(X_obs)
    else:
        X = gen_observables(lbr_, E, O={})

    return X

if __name__ == '__main__':

    PATH = 'data/'

    import mpi4py.MPI as MPI
    import tqdm as tqdm
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    values = []
    num_simulations = size

    test = True

    # eperimental parameters (keep fixed across simulations)
    K = 12          # number of cortical depths
    T_sim = 40      # simulation time
    L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I = 0, 1, 2, 3, 4, 5, 6, 7
    nu = 15                                             # Hz
    S_th = 902                                          # number of thalamic inputs
    P_th = np.array([0.0983, 0.0619, 0.0512, 0.0196])   # connection probability thalamic inputs
    E = {'T': T_sim, 'TR': 2, 'K': K, 
            'stimulations': 
                               [{'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[0],  'target': [L4E]},
                                {'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[1],  'target': [L4I]},
                                {'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[2],  'target': [L6E]},
                                {'onset': 5, 'duration': 10, 'amplitude': nu * S_th * P_th[3],  'target': [L6I]}]
            }

    theta = create_theta(num_simulations, components=['NVC', 'LBR'], parameters=[['c1', 'c2', 'c3'],['V0t', 'V0t_p']])

    params_per_worker = np.array_split(theta, comm.Get_size())

    num_simulations_per_worker = int(num_simulations / size)
    worker_params = params_per_worker[rank]

    X = []
    for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
        theta_i = worker_params[i]
        X_i = worker(E, O={}, theta=theta_i, test=test)
        X_i.update(theta_i)
        X.append(X_i)

    # gather results
    X = comm.allgather(X)

    if rank == 0:
        X = np.ravel(X)
        X = np.array(X)

        import pylab as plt
        plt.figure()
        vmin, vmax = np.min(X[0]['lbr']), np.max(X[0]['lbr'])
        vall = np.max(np.abs([vmin, vmax]))
        plt.imshow(X[0]['lbr'].T, aspect='auto', interpolation='nearest', cmap='PiYG', vmin=-vall, vmax=vall)
        plt.xticks(np.linspace(0, int(T_sim/E['TR'])-1, 4), np.linspace(0, T_sim, 4).astype(int))
        plt.xlabel('time (TR)')
        plt.ylabel('cortical depth (K)')
        plt.colorbar()
        plt.savefig('svg/imshow_lbr_ff.svg')

        # imshow the neuro signal
        plt.figure()
        vmin, vmax = np.min(X[0]['neuro']), np.max(X[0]['neuro'])
        vall = np.max(np.abs([vmin, vmax]))
        plt.imshow(X[0]['neuro'].T, aspect='auto', interpolation='none', cmap='Reds', vmin=vmin, vmax=vall)
        plt.xlabel('time (TR)')
        plt.ylabel('cortical depth (K)')
        plt.colorbar()
        plt.savefig('svg/imshow_neuro_ff.svg')

        # plot peak of syn_signal
        # plt.figure(figsize=(4, 4))
        # plt.barh(y=np.arange(K), width=X[0]['syn_signal'][100000,:], color='#276419ff', lw=3)
        # plt.yticks(np.arange(0, K, 4), np.arange(0, K, 4))
        # plt.ylabel('cortical depth (K)')
        # plt.xlabel('peak $I_{syn}$ [pA]')
        # plt.gca().invert_yaxis()
        # plt.savefig('svg/peak_syn.svg')


        # imshow the cbf signal
        plt.figure()
        vmin, vmax = np.min(X[0]['cbf']), np.max(X[0]['cbf'])
        vall = np.max(np.abs([vmin, vmax]))
        plt.imshow(X[0]['cbf'].T, aspect='auto', interpolation='none', cmap='PiYG', vmin=vmin, vmax=vall)
        plt.xlabel('time (TR)')
        plt.ylabel('cortical depth (K)')
        plt.colorbar()
        plt.savefig('svg/imshow_cbf_ff.svg')

        # plot the peaks per layer
        # plt.figure(figsize=(4, 4))
        # plt.plot(X[0]['peak_Ampl'][0], np.arange(K), color='#276419ff', lw=3)
        # plt.yticks(np.arange(0, K, 3), np.arange(0, K, 3))
        # plt.ylim(0, K-1)
        # plt.xlim(np.min(X[0]['peak_Ampl'][0]))
        # plt.xlabel('peak amplitude')
        # plt.ylabel('cortical depth (K)')
        # plt.savefig('svg/plot_peak_Ampl_ff.svg')

        if test:
            import IPython; IPython.embed()
