import numpy as np
import pylab as plt


def gen_stats(BOLD, dt):

    num_stats = 14

    if np.any(np.isnan(BOLD)):
        return np.ones(num_stats) * np.nan
    
    else:

        # get moments of BOLD signal
        mean = np.mean(BOLD)
        std = np.std(BOLD)
        skew = np.mean((BOLD - mean)**3) / std**3
        kurt = np.mean((BOLD - mean)**4) / std**4


        # find max and min values
        max_val = np.max(BOLD)
        min_val = np.min(BOLD)

        max_pos = np.argmax(BOLD) * dt
        min_pos = np.argmin(BOLD) * dt

        # find max and min slopes
        dBOLD = np.diff(BOLD)
        max_slope = np.max(dBOLD)
        min_slope = np.min(dBOLD)

        max_slope_pos = np.argmax(dBOLD) * dt 
        min_slope_pos = np.argmin(dBOLD) * dt

        # find positive area under BOLD response
        positive_area = np.sum(BOLD[BOLD > 0]) * dt

        # find negative area under BOLD response
        negative_area = np.sum(BOLD[BOLD < 0]) * dt

        stats = np.array([
            mean,
            std,
            skew,
            kurt,
            max_val,
            min_val,
            max_pos,
            min_pos,
            max_slope,
            min_slope,
            max_slope_pos,
            min_slope_pos,
            positive_area,
            negative_area,
        ])

        return stats


if __name__=='__main__':

    # load BOLD data
    BOLD = np.load('PDCM_BOLD.npy', allow_pickle=True)

    dt = 0.001
    stats = gen_stats(BOLD[:,0], dt)

    # plot BOLD signal
    plt.figure()
    plt.plot(BOLD)
    plt.show()

    import IPython; IPython.embed()









# WIP
# def gen_stats(BOLD, dt):

    # num_stats = 8

    # if np.any(np.isnan(BOLD)):
    #     return np.ones(num_stats) * np.nan
    
    # else:
    #     # compute summary statistics from BOLD signal (e.g. peak and undershoot)
    #     from scipy.signal import find_peaks

    #     # find peaks and valleys
    #     peaks, _   = find_peaks( BOLD[:, 0])
    #     valleys, _ = find_peaks(-BOLD[:, 0])

    #     # possible peaks and valleys
    #     # peaks: main peak, post stimulus peak
    #     # valleys: initial dip, steady state, undershoot

    #     #           peak   poststim_peak  initial_dip  steady_state  undershoot
    #     presence = [False, False,         False,       False,        False     ]

    #     initial_dip_amp = 0
    #     peak_amp = 0
    #     steadystate_amp = 0
    #     poststim_peak_amp = 0
    #     undershoot_amp = 0

    #     # check if peaks is found
    #     if peaks.shape[0] == 1:
    #         presence[0] = True  # main peak
    #         peak_amp = BOLD[peaks[0], 0]
    #     if peaks.shape[0] == 2:
    #         presence[0] = True  # main peak
    #         presence[1] = True  # post stimulus peak
    #         peak_amp = BOLD[peaks[0], 0]
    #         poststim_peak_amp = BOLD[peaks[1], 0]

    #     # check if valleys is found
    #     if valleys.shape[0] == 3:
    #         presence[2] = True  # initial dip
    #         presence[3] = True  # steady state
    #         presence[4] = True  # undershoot
    #         initial_dip_amp = BOLD[valleys[0], 0]
    #         steadystate_amp = BOLD[valleys[1], 0]
    #         undershoot_amp = BOLD[valleys[2], 0]
    #     if valleys.shape[0] == 2:
    #         if valleys[0] < peaks[0]:
    #             presence[2] = True  # initial dip
    #             initial_dip_amp = BOLD[valleys[0], 0]
    #         else:
    #             presence[3] = True  # steady state
    #             steadystate_amp = BOLD[valleys[0], 0]
    #             undershoot_amp  = BOLD[valleys[1], 0]
    #     if valleys.shape[0] == 1:
    #         if valleys[0] < peaks[0]:
    #             presence[2] = True  # initial dip
    #             initial_dip_amp = BOLD[valleys[0], 0]
    #         elif BOLD[valleys[0]] > 0:
    #             presence[3] = True  # steady state
    #             steadystate_amp = BOLD[valleys[0], 0]
    #         else:
    #             presence[4] = True  # undershoot
    #             undershoot_amp  = BOLD[valleys[0], 0]
        

    #     stats = np.array([
    #         initial_dip_amp,
    #         peak_amp,
    #         steadystate_amp,
    #         poststim_peak_amp,
    #         undershoot_amp,
    #     ])

    #     return stats