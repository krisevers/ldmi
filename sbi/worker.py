import numpy as np

import simulators.DCM_model as DCM_model
import simulators.NVC_model as NVC_model
import simulators.LBR_model as LBR_model


"""
Selection of parameter worker functions for DCM > NVC > LBR models
"""

def worker(K=3, T=20, DCM=None, NVC=None, LBR=None):
    """
    Wrapper function for DCM, NVC and LBR models
    """

    # prepare parameters for inference models
    if DCM is not None:
        DCM_params = {'K': K, 'M': 2*K, 'dt': 0.001, 'C': np.eye(K*2), 'T': T}
        # connectivity
        _sigma   = -3
        _mu      = 1.5
        _lambda_ = 0.2
        W = np.array([[_sigma, -_mu],           # within layer connectivity
                      [_lambda_, -_lambda_]])
        Z = np.zeros((2, 2))                    # between layer connectivity
        DCM_params['W'] = np.tile(Z, (K, K))
        for i in range(K):
            DCM_params['W'][i*2:(i+1)*2, i*2:(i+1)*2] = W
        # external input
        U = {}
        dur = DCM['dur'] / DCM_params['dt']  						# Stimulus duration (in second, e.g. 2 sec) ... dt - refers to integration step
        onset = int(3 / DCM_params['dt'])  						    # Stimulus onset time (in seconds)
        offset = int(onset + dur) 			 			            # Stimulus offset time (in seconds)
        U['u'] = np.zeros((int(T / DCM_params['dt']), K*2))         # Matrix with input vectors to the neuronal model (one column per depth)
        U['u'][onset:offset, 0] = DCM['U_1']
        U['u'][onset:offset, 2] = DCM['U_2']
        U['u'][onset:offset, 4] = DCM['U_3'] 
    else: 
        # default parameters
        DCM_params = {'K': K, 'M': 2*K, 'dt': 0.001, 'C': np.eye(K*2), 'T': T}
        # connectivityfor i in range(K):
        _sigma   = -3
        _mu      = 1.5
        _lambda_ = 0.2
        W = np.array([[_sigma, -_mu],           # within layer connectivity
                      [_lambda_, -_lambda_]])
        Z = np.zeros((2, 2))                    # between layer connectivity
        DCM_params['W'] = np.tile(Z, (K, K))
        for i in range(K):
            DCM_params['W'][i*2:(i+1)*2, i*2:(i+1)*2] = W
        # external input
        U = {}
        dur = 2 / DCM_params['dt']  						        # Stimulus duration (in second, e.g. 2 sec) ... dt - refers to integration step
        onset = int(3 / DCM_params['dt'])  						    # Stimulus onset time (in seconds)
        offset = int(onset + dur) 			 			            # Stimulus offset time (in seconds)
        U['u'] = np.zeros((int(T / DCM_params['dt']), K*2))         # Matrix with input vectors to the neuronal model (one column per depth)
        for i in range(K):
            U['u'][onset:offset, i*2] = 1

    if NVC is not None:
        NVC_params = NVC
    else:
        # default parameters
        NVC_params = {'K': K, 'dt': 0.001, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6, 'T': T}

    if LBR is not None:
        LBR_params = LBR
    else:
        # default parameters
        LBR_params = {'K': K, 'dt': 0.01, 'T': T}
        LBR_params = LBR_model.LBR_parameters(K, LBR_params)
        LBR_params['alpha_v']  = 0.35
        LBR_params['alpha_d']  = 0.2
        LBR_params['tau_d_de'] = 30

    # simulate
    neuro = DCM_model.DCM_sim(U, DCM_params)

    # TODO: N2K neuro > [T, M], cbf > [T, K], lbr > [T, K]    

    cbf = NVC_model.NVC_sim(neuro[:, ::2], NVC_params)

    new_dt = 0.01
    old_dt = DCM_params['dt']
    lbr_dt = new_dt/old_dt
    cbf_ = cbf[::int(lbr_dt)]

    LBR_params['dt'] = new_dt

    lbr, lbrpial, Y = LBR_model.LBR_sim(cbf_, LBR_params)

    # return {'neuro': neuro, 'cbf': cbf, 'lbr': lbr}


    # check for nan values
    if np.isnan(lbr).any() or np.isnan(cbf).any() or np.isnan(neuro).any():
        peak_Posi = np.nan
        peak_Ampl = np.nan
        peak_Area = np.nan
        unde_Posi = np.nan
        unde_Ampl = np.nan
        unde_Area = np.nan
        up_Slope = np.nan
        down_Slope = np.nan
    else:
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

    return {'peak_Posi':  peak_Posi, 
            'peak_Ampl':  peak_Ampl, 
            'peak_Area':  peak_Area, 
            'unde_Posi':  unde_Posi, 
            'unde_Ampl':  unde_Ampl, 
            'unde_Area':  unde_Area, 
            'up_Slope':   up_Slope, 
            'down_Slope': down_Slope}

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

    num_simulations = 100


    K = 3           # number of cortical depths
    T_sim = 30      # simulation time

    DCM_theta = np.transpose([
        np.random.uniform(0, 1,  size=num_simulations),  # U_1
        np.random.uniform(0, 1,  size=num_simulations),  # U_2
        np.random.uniform(0, 1,  size=num_simulations),  # U_3
        np.random.uniform(1, 10, size=num_simulations)   # stimulus duration
    ])

    NVC_theta = np.transpose([
        np.ones(num_simulations) * 0.6,  # c1
        np.ones(num_simulations) * 1.5,  # c2
        np.ones(num_simulations) * 0.6   # c3
    ])

    
    LBR_theta = np.transpose([
        np.full(num_simulations, None)
    ])

    DCM_paras_per_worker = np.array_split(DCM_theta, comm.Get_size())
    NVC_paras_per_worker = np.array_split(NVC_theta, comm.Get_size())
    LBR_paras_per_worker = np.array_split(LBR_theta, comm.Get_size())

    num_simulations_per_worker = int(num_simulations / size)
    DCM_workers_params = DCM_paras_per_worker[rank]
    NVC_workers_params = NVC_paras_per_worker[rank]
    LBR_workers_params = LBR_paras_per_worker[rank]

    X = []
    for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
        DCM = {'U': DCM_workers_params[i, :3], 'dur': DCM_workers_params[i, 3]}
        NVC = {'c1': NVC_workers_params[i, 0], 'c2': NVC_workers_params[i, 1], 'c3': NVC_workers_params[i, 2]}
        LBR = None
        X_i = worker(K=K, T=T_sim, DCM=DCM, NVC=NVC, LBR=None)
        X_i.update(DCM)
        X_i.update(NVC)
        X_i.update(LBR)
        X.append(X_i)

    all_X = comm.allgather(X)

    if rank == 0:
        model_name = 'DCM_NVC_LBR'

        all_X = np.reshape(all_X, (num_simulations))

        np.save(PATH + 'X.npy', all_X)