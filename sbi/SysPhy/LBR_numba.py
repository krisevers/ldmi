import numpy as np

from numba import jit

def LBRparams(K, theta=None, cmro2=None):
        # --------------------------------------------------------------------------
        # Default parameters -- provide in as list suitable for numba
        P = np.empty((39), dtype=object)

        P[0] = 0.001

        depths = np.linspace(0,100,2*K+1) # Normalized distance to the center of individual depths (in %)
        P[1] = depths[0::2]
        # LAMINAR HEMODYNAMIC MODEL:
        #--------------------------------------------------------------------------
        # Baseline physiological parameters:
        if 'V0t' not in theta:
            P[2] = 2.5  	# Total (regional) amount of CBV0 in the gray matter (in mL) [1-6]
        else:
            P[2] = theta['V0t']
        if 'V0t_p' not in theta:
            P[3] = 1  	    # Total (regional) amount of CBV0 in the pial vein (mL) [1-6]
        else:
            P[3] = theta['V0t_p']

        if 'w_v' not in theta:
            P[4] = 0.5  # CBV0 fraction of microvasculature (i.e. venules here) with respect to the total amount 
        else:
            P[4] = theta['w_v']
        P[5] = []   # CBV0 fraction across depths in venules 
        P[6] = []   # CBV0 fraction across depths in ascending veins
        if 's_v' not in theta:
            P[7] = 0    # Slope of CBV increase (decrease) in venules [0-0.3]
        else:
            P[7] = theta['s_v']
        if 's_d' not in theta:
            P[8] = 0.3  # Slope of CBV increase in ascending vein     [0-1.5]
        else:
            P[8] = theta['s_d']

        if 't0v' not in theta:
            P[9] = 1    # Transit time through microvasculature(in second)
        else:
            P[9] = theta['t0v']
        if 'E0v' not in theta:
            P[10] = 0.35 # Baseline oxygen extraction fraction in venules
        else:
            P[10] = theta['E0v']
        if 'E0d' not in theta:
            P[11] = 0.35 # Baseline oxygen extraction fraction in ascending vein
        else:
            P[11] = theta['E0d']
        if 'E0p' not in theta:
            P[12] = 0.35 # Baseline oxygen extraction fraction in pial vein
        else:
            P[12] = theta['E0p']

        # Parameters describing relative relationship between physiological variable:
        # CBF-CBV coupling (steady-state)
        if 'alpha_v' not in theta:
            P[13] = 0.3     # For venules
        else:
            P[13] = theta['alpha_v']
        if 'alpha_d' not in theta:
            P[14] = 0.2     # For ascending vein
        else:
            P[14] = theta['alpha_d']
        if 'alpha_p' not in theta:
            P[15] = 0.1     # For pial vein
        else:
            P[15] = theta['alpha_p']

        # CBF-CMRO2 coupling (steady-state)
        if 'n' not in theta:
            P[16] = 4          # n-ratio   (Ref. Buxton et al. (2004) NeuroImage)
        else:
            P[16] = theta['n']

        # CBF-CBV dynamic uncoupling
        if 'tau_v_in' not in theta:
            P[17] = 2  # For venules - inflation
        else:
            P[17] = theta['tau_v_in']
        if 'tau_v_de' not in theta:
            P[18] = 2  #             - deflation
        else:
            P[18] = theta['tau_v_de']

        if 'tau_d_in' not in theta:
            P[19] = 2  # For ascending vein - inflation
        else:
            P[19] = theta['tau_d_in']
        if 'tau_d_de' not in theta:
            P[20] = 2  #                    - deflation
        else:
            P[20] = theta['tau_d_de']

        if 'tau_p_in' not in theta:
            P[21] = 2  # For pial vein - inflation
        else:
            P[21] = theta['tau_p_in']
        if 'tau_p_de' not in theta:
            P[22] = 2  #               - deflation
        else:
            P[22] = theta['tau_p_de']

        # LAMINAR BOLD SIGNAL MODEL:
        #--------------------------------------------------------------------------
        if 'TE' not in theta:
            P[23]     = 0.028    # echo-time (in sec)
        else:
            P[23]     = theta['TE']

        # Hematocrit fraction
        if 'Hct_v' not in theta:
            P[24]  = 0.35 	# For venules, Ref. Lu et al. (2002) NeuroImage
        else:
            P[24]  = theta['Hct_v']
        if 'Hct_d' not in theta:
            P[25]  = 0.38		# For ascending vein
        else:
            P[25]  = theta['Hct_d']
        if 'Hct_p' not in theta:
            P[26]  = 0.42  	# For pial vein
        else:
            P[26]  = theta['Hct_p']


        if 'B0' not in theta:
            P[27]     = 7   					# Magnetic field strenght (in Tesla)  
        else:
            P[27]     = theta['B0']
        if 'gyro' not in theta:
            P[28]   = 2*np.pi*42.6*10**6  	# Gyromagnetic constant for Hydrogen
        else:
            P[28]   = theta['gyro']
        if 'suscep' not in theta:
            P[29] = 0.264*10**-6       	# Susceptibility difference between fully oxygenated and deoxygenated blood
        else:
            P[29] = theta['suscep']

        # Water proton density:
        if 'rho_t' not in theta:
            P[30]  = 0.89                 		    # For gray matter tissue 
        else:
            P[30]  = theta['rho_t']
        P[31]  = 0.95 - P[24]*0.22  	    # For blood (venules) Ref. Lu et al. (2002) NeuroImage
        P[32]  = 0.95 - P[25]*0.22  	    # For blood (ascending vein)
        P[33]  = 0.95 - P[26]*0.22  	    # For blood (pial vein)
        if 'rho_tp' not in theta:
            P[34] = 0.95                 		    # For gray matter tissue % CSF   
        else:
            P[34] = theta['rho_tp']

        # Relaxation rates for 7 T (in sec-1)
        if 'R2s_t' not in theta:
            P[35]  = 34  # For gray matter tissue
        else:
            P[35]  = theta['R2s_t']
        if 'R2s_v' not in theta:
            P[36]  = 80  # For blood (venules)
        else:
            P[36]  = theta['R2s_v']
        if 'R2s_d' not in theta:
            P[37]  = 85  # For blood (ascending vein)
        else:
            P[37]  = theta['R2s_d']
        if 'R2s_p' not in theta:
            P[38]  = 90  # For blood (pial vein)
        else:
            P[38]  = theta['R2s_p']

        return P

def LBRinit(K, P):

    # if len(args) < 1:
    #     cmro2 = []

    cmro2 = []

    ##
    # Hemodynamic model parameters
    #------------------------------------------------------

    # BASELINE PARAMETERS
    V0t 	= np.float128(P[2])		                # Total amount of CBV0 within GM tissue (in mL)
    V0t_p	= np.float128(P[3])	                    # Total amount of CBV0 in pial vein (in mL)

    w_v		= np.float128(P[4])		                # Fraction of CBV0 in venules with respect to the total
    w_d  	= 1-w_v			                        # Fraction of CBV0 in ascending vein with respect to the total

    s_v 	= np.float128(P[7])		                # Slope of CBV0 increase towards the surface in venules
    s_d  	= np.float128(P[8])         		    # Slope of CBV0 increase towards the surface in ascending veins

    # Depth-specific CBV0
    x_v     = (10+s_v*np.flipud(P[1])).astype(np.float128) # Possibility to define linear increase (default s_v = 0)
    x_v     = x_v/np.sum(x_v)                      # Fraction of CBV0 across depths in venules 

    x_d     = 10+s_d*np.flipud(P[1]).astype(np.float128) # Possibility to define linear increase 
    x_d     = x_d/np.sum(x_d)          # Fraction of CBV0 across depths in venules 

    V0v      = V0t*w_v*x_v              # CBV0 in venules
    V0d      = V0t*w_d*x_d              # CBV0 in ascending vein
    V0p      = V0t_p                    # CBV0 in pial vein

    # Transit time through venules (or microvasculature in general)
    t0v = (np.ones(K)*P[9]).astype(np.float128)

    # Depth-specific baseline CBF
    F0v = V0v/t0v
    F0d = np.flipud(np.cumsum(np.flipud(F0v)))
    F0p = F0d[1]

    # Depth-specific transit time
    t0v = V0v/F0v
    t0d = V0d/F0d
    t0p = V0p/F0p

    # Total mean transit time
    tt0v = np.mean(t0v)
    tt0d = np.mean(np.cumsum(t0d))
    tt0  = tt0v + tt0d

    # Baseline oxygen extraction fraction
    E0v        = (np.ones(K)*P[10]).astype(np.float128)
    E0d        = (np.ones(K)*P[11]).astype(np.float128)
    E0p        = np.float128(P[12])


    # PARAMETERS DESCRIBING RELATIVE RELATIONSHIPS BETWEEN PHYSIOLOGICAL VARIABLES:
    # n-ratio (= (cbf-1)./(cmro2-1)). Not used if cmro2 response is directly specified as an input
    n      = (np.ones(K)*P[16]).astype(np.float128)                # Default

    # Grubb's exponent alpha (i.e CBF-CBV steady-state relationship)
    alpha_v    = (np.ones(K)*P[13]).astype(np.float128)  	# Default
    alpha_d    = (np.ones(K)*P[14]).astype(np.float128)  	# Default
    alpha_p    = np.float128(P[15])      			        # For pial vein

    # CBF-CBV uncoupling (tau) during inflation and deflation:
    tau_v_in  = (np.ones(K)*P[17]).astype(np.float128) 	    # Default  
    tau_v_de  = (np.ones(K)*P[18]).astype(np.float128)  	# Default  
    tau_d_in  = (np.ones(K)*P[19]).astype(np.float128)   	# Default  
    tau_d_de  = (np.ones(K)*P[20]).astype(np.float128)  	# Default
    tau_p_in      = np.float128(P[21])       		            # For pial vein (inflation)
    tau_p_de      = np.float128(P[22])       		            # For pial vein (deflation)



    ##
    # Parameters for laminar BOLD signal equation (for 7 T field strenght):
    #------------------------------------------------------
    # Baseline CBV in fraction with respect to GM tissue
    V0vq = np.float128(V0v/100*K)
    V0dq = np.float128(V0d/100*K)
    V0pq = np.float128(V0p/100*K)

    TE     = np.float128(P[23])	 	 # echo-time (sec) 

    Hct_v  = np.float128(P[24])	 # Hematocrit fraction
    Hct_d  = np.float128(P[25])
    Hct_p  = np.float128(P[26])
    B0     = np.float128(P[27])   	 # Field strenght        
    gyro   = np.float128(P[28])      # Gyromagnetic constant 
    suscep = np.float128(P[29])    # Susceptibility difference

    nu0v   = np.float128(suscep*gyro*Hct_v*B0)
    nu0d   = np.float128(suscep*gyro*Hct_d*B0)
    nu0p   = np.float128(suscep*gyro*Hct_p*B0)

    # Water proton density 
    rho_t  = np.float128(P[30])  # In GM tissue
    rho_v  = np.float128(P[31])  # In blood (venules) Ref. Lu et al. (2002) NeuroImage
    rho_d  = np.float128(P[32])  # In blood (ascening vein) 
    rho_p  = np.float128(P[33])  # In blood (pial vein) 
    rho_tp = np.float128(P[34]) # In in tissue and CSF 

    # Relaxation rates (in sec-1):
    R2s_t  = (np.ones(K)*P[35]).astype(np.float128)   	# (sec-1)
    R2s_v  = (np.ones(K)*P[36]).astype(np.float128)  	# (sec-1) 
    R2s_d  = (np.ones(K)*P[37]).astype(np.float128) 	# (sec-1)  
    R2s_p  = np.float128(P[38])         			# For pial vein 

    # (Baseline) Intra-to-extra-vascular signal ratio
    ep_v   = (rho_v/rho_t*np.exp(-TE*R2s_v)/np.exp(-TE*R2s_t)).astype(np.float128) 	# For venules
    ep_d   = (rho_d/rho_t*np.exp(-TE*R2s_d)/np.exp(-TE*R2s_t)).astype(np.float128)	# For ascending vein
    ep_p   = (rho_p/rho_tp*np.exp(-TE*R2s_p)/np.exp(-TE*R2s_t)).astype(np.float128)	# For pial vein 

    # Slope of change in R2* of blood with change in extraction fration during activation 
    r0v    = np.float128(228)	 # For venules   
    r0d    = np.float128(232)    # For ascending vein
    r0p    = np.float128(236)    # For pial vein

    H0     = (1/(1 - V0vq - V0dq + ep_v*V0vq + ep_d*V0dq)).astype(np.float128)	# constant in front
    H0p    = (1/(1 - V0pq + ep_p*V0pq)).astype(np.float128)

    k1v     = np.float128(4.3*nu0v*E0v*TE)
    k2v     = np.float128(ep_v*r0v*E0v*TE)
    k3v     = np.float128(1 - ep_v)

    k1d     = np.float128(4.3*nu0d*E0d*TE)
    k2d     = np.float128(ep_v*r0d*E0d*TE)
    k3d     = np.float128(1 - ep_d)

    k1p     = np.float128(4.3*nu0p*E0p*TE)
    k2p     = np.float128(ep_p*r0p*E0p*TE)
    k3p     = np.float128(1 - ep_p)


    ##
    # Initial conditions
    #------------------------------------------------------
    Xk       = np.zeros((K,4)).astype(np.float128)
    Xp       = np.zeros((2)).astype(np.float128)

    yk       = Xk
    yp       = Xp

    f_d      = np.ones(K).astype(np.float128)
    dv_d     = np.ones(K).astype(np.float128)
    dHb_d    = np.ones(K).astype(np.float128)

    tau_v    = tau_v_in
    tau_d    = tau_d_in
    tau_p    = tau_p_in

    # integration step
    dt = P[0]
    t_steps = np.shape(F)[0]

    LBR       = np.zeros((t_steps,K)).astype(np.float128)
    LBRpial   = np.zeros((t_steps,K)).astype(np.float128)

    Y = np.empty((23), dtype=object)
    Y[0] = np.zeros((t_steps, K))   # f_a
    Y[1] = np.zeros((t_steps, K))   # m_v
    Y[2] = np.zeros((t_steps, K))   # q_v
    Y[3] = np.zeros((t_steps, K))   # q_d
    Y[4] = np.zeros((t_steps))      # q_p
    
    Y[5] = np.zeros((t_steps, K))    # v_v
    Y[6] = np.zeros((t_steps, K))    # v_d
    Y[7] = np.zeros((t_steps, K))    # v_p

    return [LBR, LBRpial, Y, Xk, Xp, yk, yp, f_d, dv_d, dHb_d, tau_v, tau_d, tau_p, dt, t_steps, cmro2, H0, H0p, k1v, k2v, k3v, k1d, k2d, k3d, k1p, k2p, k3p]


@jit(nopython=True)
def LBRsim(F, K, P, A):

    LBR, LBRpial, Y, Xk, Xp, yk, yp, f_d, dv_d, dHb_d, tau_v, tau_d, tau_p, dt, t_steps, cmro2, H0, H0p, k1v, k2v, k3v, k1d, k2d, k3d, k1p, k2p, k3p = A
    #
    # Simulation
    #------------------------------------------------------            
    for t in range(t_steps):
        Xk      = np.exp(Xk)    # log-normal transformation (Stephan et al.(2008), NeuroImage)
        Xp      = np.exp(Xp)
        
        # model input (laminar CBF response):
        f_a = (F[t,:].T).astype(np.float128)
        
        # VENULES COMPARTMENTS:
        #--------------------------------------------------------------------------
        # blood outflow from venules compartment
        if np.sum(alpha_v)>0:
            f_v     = (V0v*Xk[:,0]**(1/alpha_v) + F0v*tau_v*f_a)/(V0v+F0v*tau_v)
        else:
            f_v     = f_a
        
        # change in blood volume in venules:
        dv_v        = (f_a - f_v)/t0v
        # change in oxygen matabolims (CMRO2)
        if len(cmro2) == 0:
            m        = (f_a + n-1)/n  # (if not specified directly)
        else:
            m        = cmro2[t,:].T

        # change in deoxyhemoglobin content venules:
        dHb_v        = (m - f_v*Xk[:,1]/Xk[:,0])/t0v

        # ASCENDING VEIN COMPARTMENTS:
        #--------------------------------------------------------------------------    
        # blood outflow from Kth depth of ascending vein compartment (deepest depth):
        if alpha_d[-1]>0:
            f_d[-1]  = (V0d[-1]*Xk[-1,2]**(1/alpha_d[-1]) + tau_d[-1]*f_v[-1]*F0v[-1])/(V0d[-1]+F0d[-1]*tau_d[-1])
        else:
            f_d[-1]  = f_v[-1]*F0v[-1]/F0d[-1]

        # changes in blood volume and deoxyhemoglobin in ascending vein (deepest depth):
        dv_d[-1]     = (f_v[-1] - f_d[-1])/t0d[-1]
        dHb_d[-1]    = (f_v[-1]*Xk[-1,1]/Xk[-1,0] - f_d[-1]*Xk[-1,3]/Xk[-1,2])/t0d[-1]
        
        # blood outflow from other comparments of ascending vein:
        for i in range(K-2, -1, -1):
            if alpha_d[i]>0:
                f_d[i]     = (V0d[i]*Xk[i,2]**(1/alpha_d[i]) + tau_d[i]*(f_v[i]*F0v[i]+f_d[i+1]*F0d[i+1]))/(V0d[i]+F0d[i]*tau_d[i])
            else:
                f_d[i]     = f_v[i]*F0v[i]/F0d[i]+f_d[i+1]*F0d[i+1]/F0d[i]
            
            # changes in blood volume and deoxyhemoglobin in ascending vein:
            dv_d[i]    = (f_v[i]*F0v[i]/F0d[i] + f_d[i+1]*F0d[i+1]/F0d[i] - f_d[i])/t0d[i]
            dHb_d[i]   = (f_v[i]*F0v[i]/F0d[i]*Xk[i,1]/Xk[i,0] + f_d[i+1]*F0d[i+1]/F0d[i]*Xk[i+1,3]/Xk[i+1,2] - f_d[i]*Xk[i,3]/Xk[i,2])/t0d[i]

        
        # PIAL VEIN COMPARTMENT:
        #--------------------------------------------------------------------------    

        # blood outflow from pial vein:
        if alpha_p>0:
            f_p     = (V0p*Xp[0]**(1/alpha_p) + F0p*tau_p*f_d[0])/(V0p+F0p*tau_p)
        else:
            f_p     = f_d[0]
        
        # changes in blood volume and deoxyhemoglobin in pial vein:
        dv_p  = (f_d[0] - f_p)/t0p
        dHb_p = (f_d[0]*Xk[0,3]/Xk[0,2] - f_p*Xp[1]/Xp[0])/t0p
        
        
        # Intergrated changes to previous time point
        yk[:,0]  = yk[:,0] + dt*(dv_v/Xk[:,0])
        yk[:,1]  = yk[:,1] + dt*(dHb_v/Xk[:,1])
        yk[:,2]  = yk[:,2] + dt*(dv_d/Xk[:,2])
        yk[:,3]  = yk[:,3] + dt*(dHb_d/Xk[:,3])
        
        yp[0]  = yp[0] + dt*(dv_p/Xp[0])
        yp[1]  = yp[1] + dt*(dHb_p/Xp[1])

        Xk        = yk
        Xp        = yp
        
        tau_v     = tau_v_in
        tau_d     = tau_d_in
        tau_p     = tau_p_in

        # check for deflation (negative derivative)
        tau_v[dv_v<0]  = tau_v_de[dv_v<0]
        tau_d[dv_d<0]  = tau_d_de[dv_d<0]
        if dv_p<0:
            tau_p  = tau_p_de
        
        # venules:
        m_v  = m
        v_v  = np.exp(yk[:,0])		# log-normal transformation
        q_v  = np.exp(yk[:,1])
        # draining vein:
        v_d  = np.exp(yk[:,2])
        q_d  = np.exp(yk[:,3])
        # pail vein:
        v_p  = np.exp(yp[0])
        q_p  = np.exp(yp[1])
        
        # save physiological variable:
        # Y[0][t,:] = f_a
        # Y[1][t,:] = m_v
        # Y[2][t,:] = q_v
        # Y[3][t,:] = q_d
        # Y[4][t]   = q_p

        # Y[5][t,:] = v_v
        # Y[6][t,:] = v_d
        # Y[7][t,:] = v_p

        
        
        LBR[t,:] = H0*((1-V0vq-V0dq)*(k1v*V0vq*(1-q_v) +k1d*V0dq*(1-q_d)) + 
                                        + k2v*V0vq*(1-q_v/v_v) + k2d*V0dq*(1-q_d/v_d) +
                                        + k3v*V0vq*(1-v_v)     + k3d*V0dq*(1-v_d))*100
        
        
        LBRpial[t,:] = H0p*((1-V0pq)*(k1p*V0pq*(1-q_p)) + k2p*V0pq*(1-q_p/v_p) +
                                                        k3p*V0pq*(1-v_p))*100

    # save baseline physiological parameters
    # Y[8]   = F0v
    # Y[9]   = F0d
    # Y[10]  = F0p

    # Y[11]  = V0v
    # Y[12]  = V0d
    # Y[13]  = V0p

    # Y[14]  = V0vq
    # Y[15]  = V0dq
    # Y[16]  = V0pq    

    # Y[17]  = t0v
    # Y[18]  = t0d
    # Y[19]  = t0p
    # Y[20]  = tt0v
    # Y[21]  = tt0d
    # Y[22]  = tt0
    
    Y = 0

    return LBR, LBRpial, Y