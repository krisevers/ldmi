import numpy as np

from simulators.DMF import DMF
from simulators.NVC import NVC
from simulators.LBR import LBR

from utils import get_N2K, gen_input, syn_to_neuro, gen_observables, create_theta


"""
Selection of parameter worker functions for DCM > NVC > LBR models
"""

def worker(E, O, theta=None):
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

    # convert synaptic activity to neural response
    idx_baseline = int(1/DMF_model.P['dt'])  # ms
    neuro = syn_to_neuro(syn_signal, E['K'], E_scale=20, I_scale=0, baseline=idx_baseline)  # [T, K]

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

    # TODO: add optional obersevables
    # X = gen_observables(lbr_, E, O={})
    X = {'lbr': lbr_}

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

    # eperimental parameters (keep fixed across simulations)
    K = 12          # number of cortical depths
    T_sim = 80     # simulation time
    L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I = 0, 1, 2, 3, 4, 5, 6, 7
    E = {'T': T_sim, 'TR': 2, 'K': K, 
         'stimulations': 
                            [{'onset': 5,  'duration': 5, 'amplitude': 20,  'target': [L23E]},
                             {'onset': 25, 'duration': 5, 'amplitude': 20,  'target': [L4E]},
                             {'onset': 45, 'duration': 5, 'amplitude': 20,  'target': [L5E]},
                             {'onset': 65, 'duration': 5, 'amplitude': 20,  'target': [L6E]}]
         }

    theta = create_theta(num_simulations, components=['NVC', 'LBR'], parameters=[['c1', 'c2', 'c3'],['V0t', 'V0t_p']])

    params_per_worker = np.array_split(theta, comm.Get_size())

    num_simulations_per_worker = int(num_simulations / size)
    worker_params = params_per_worker[rank]

    X = []
    for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
        theta_i = worker_params[i]
        X_i = worker(E, O={}, theta=theta_i)
        X_i.update(theta_i)
        X.append(X_i)

    # gather results
    X = comm.allgather(X)

    if rank == 0:
        X = np.ravel(X)
        X = np.array(X)

        import pylab as plt
        plt.figure()
        plt.imshow(X[0]['lbr'].T, aspect='auto', interpolation='none')
        plt.xlabel('time (TR)')
        plt.ylabel('cortical depth (K)')
        plt.colorbar()
        plt.savefig('png/test/imshow_lbr_ex{}.png'.format(0))

        import IPython; IPython.embed()


















    # if rank == 0:
    #     model_name = 'DMF_test'

    #     import pylab as plt
    #     import matplotlib.colors as colors
    #     import matplotlib.cbook as cbook
    #     from matplotlib import cm

    #     N2K, TH = get_N2K(K)
    #     TH_, ind = np.unique(TH, return_index=True)
    #     TH_ = TH_[np.argsort(ind)]

    #     for tr in range(len(X)):
    #         cm_neuro = plt.cm.Spectral(np.linspace(0, 1, K))
    #         cm_cbf   = plt.cm.Spectral(np.linspace(0, 1, K))
    #         cm_lbr   = plt.cm.Spectral(np.linspace(0, 1, K))
    #         plt.figure(figsize=(10, 10))
    #         plt.subplot(3, 1, 1)
    #         for i in range(K):
    #             plt.plot(X[tr]['neuro'][:,i], color=cm_neuro[i])
    #         plt.subplot(3, 1, 2)
    #         for i in range(K):
    #             plt.plot(X[tr]['cbf'][:, i], color=cm_cbf[i])
    #         plt.subplot(3, 1, 3)
    #         for i in range(K):
    #             plt.plot(X[tr]['lbr'][:, i], color=cm_lbr[i])
    #         plt.savefig('png/test/plot_neuro_cbf_lbr_ex{}.png'.format(tr))

    #         layers = ['L23', 'L4', 'L5', 'L6']
    #         layer_pos = [2, 7, 11, 14]
    #         plt.text(0, 2, layers[0], color='black', fontsize=20)
    #         plt.text(0, 7, layers[1], color='black', fontsize=20)
    #         plt.text(0, 11, layers[2], color='black', fontsize=20)
    #         plt.text(0, 14, layers[3], color='black', fontsize=20)

    #         plt.figure(figsize=(10, 10))
    #         plt.subplot(3, 1, 1)
    #         plt.imshow(X[tr]['neuro'].T, cmap='Spectral', aspect='auto', interpolation='none')
    #         plt.colorbar()
    #         for i in range(4):
    #             plt.text(1/1e-4, layer_pos[i], layers[i], color='black')
    #         plt.subplot(3, 1, 2)
    #         plt.imshow(X[tr]['cbf'].T, cmap='Spectral', aspect='auto', interpolation='none')
    #         plt.colorbar()
    #         for i in range(4):
    #             plt.text(1/1e-2, layer_pos[i], layers[i], color='black')
    #         plt.subplot(3, 1, 3)
    #         plt.imshow(X[tr]['lbr'].T, cmap='Spectral', aspect='auto', interpolation='none')
    #         plt.colorbar()
    #         for i in range(4):
    #             plt.text(1/1e-2, layer_pos[i], layers[i], color='black')
    #         plt.tight_layout()
    #         plt.savefig('png/test/imshow_neuro_cbf_lbr_ex{}.png'.format(tr))


        # import IPython
        # IPython.embed()
