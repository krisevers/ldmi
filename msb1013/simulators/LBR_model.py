import numpy as np

def LBR_sim(cbf, P, *args):
    '''
    INPUTS:
        cbf - matrix defining laminar cerebral blood flow (CBF) response, [time, depth]
        P - structure of model parameters
        cmro2 - matrix defining the laminar changes in oxygen metabolism (CMRO2), [time, depth]

	OUTPUTS:
    LBR - matrix containing laminar BOLD responses in percent signal change [time, depth]
    Y - structure with all baseline and relative physiological variables underlying BOLD response
    LBRpial - BOLD response of the pial vein in percent signal change (0th depth) [time, 1]

    AUTHOR: Martin Havlicek, 5 August, 2019
    '''

    if len(args) < 1:
        cmro2 = []

	##
	# Hemodynamic model parameters
	#------------------------------------------------------
    K = int(P['K'])		# Number of depths

	# BASELINE PARAMETERS
    V0t 	= np.float128(P['V0t'])		# Total amount of CBV0 within GM tissue (in mL)
    V0t_p	= np.float128(P['V0t_p'])	# Total amount of CBV0 in pial vein (in mL)

    w_v		= np.float128(P['w_v'])		# Fraction of CBV0 in venules with respect to the total
    w_d  	= 1-w_v			            # Fraction of CBV0 in ascending vein with respect to the total

    s_v 	= np.float128(P['s_v'])		# Slope of CBV0 increase towards the surface in venules
    s_d  	= np.float128(P['s_d'])		# Slope of CBV0 increase towards the surface in ascending veins

    # Depth-specific CBV0
    if len(P['x_v']) == K:                  # For venules
        x_v  = P['x_v'].astype(np.float128) # Depth-specific fractions defined by user
    else:
        x_v  = (10+s_v*np.flipud(P['l'])).astype(np.float128) # Possibility to define linear increase (default s_v = 0)
    x_v      = x_v/np.sum(x_v)          # Fraction of CBV0 across depths in venules 

    if len(P['x_v']) == K:                  # For ascending vein
        x_d  = P['x_d'].astype(np.float128) # Depth-specific fractions defined by user
    else:
        x_d  = 10+s_d*np.flipud(P['l']).astype(np.float128) # Possibility to define linear increase 
    x_d      = x_d/np.sum(x_d)          # Fraction of CBV0 across depths in venules 

    V0v      = V0t*w_v*x_v              # CBV0 in venules
    V0d      = V0t*w_d*x_d              # CBV0 in ascending vein
    V0p      = V0t_p                    # CBV0 in pial vein

	# Transit time through venules (or microvasculature in general)
    if hasattr(P['t0v'], '__len__'):
        t0v = P['t0v'].astype(np.float128)
    else:
        t0v = (np.ones(K)*P['t0v']).astype(np.float128)

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
    if hasattr(P['E0v'], '__len__'):
        E0v        = P['E0v'].astype(np.float128)     # depth-specific defined by user
    else:
        E0v        = (np.ones(K)*P['E0v']).astype(np.float128)
    if hasattr(P['E0d'], '__len__'):
        E0d        = P['E0d'].astype(np.float128)      # depth-specific defined by user
    else:
        E0d        = (np.ones(K)*P['E0d']).astype(np.float128)
    E0p        = np.float128(P['E0p'])


	# PARAMETERS DESCRIBING RELATIVE RELATIONSHIPS BETWEEN PHYSIOLOGICAL VARIABLES:
	# n-ratio (= (cbf-1)./(cmro2-1)). Not used if cmro2 response is directly specified as an input
    if hasattr(P['n'], '__len__'):      # For venules (microvasculature)
        n      = np.float128(P['n'])    # Depth-specific defined by user
    else:
        n      = (np.ones(K)*P['n']).astype(np.float128)                # Default

	# Grubb's exponent alpha (i.e CBF-CBV steady-state relationship)
    if hasattr(P['alpha_v'], '__len__'):		                        # For venules
        alpha_v    = P['alpha_v'].astype(np.float128)       		    # Depth-specific defined by user 
    else:
        alpha_v    = (np.ones(K)*P['alpha_v']).astype(np.float128)  	# Default
    if hasattr(P['alpha_d'], '__len__'):		                        # For ascending vein
        alpha_d    = P['alpha_d'].astype(np.float128)             	    # Depth-specific defined by user  
    else:
        alpha_d    = (np.ones(K)*P['alpha_d']).astype(np.float128)  	# Default
    alpha_p        = np.float128(P['alpha_p'])      			        # For pial vein

	# CBF-CBV uncoupling (tau) during inflation and deflation:
    if hasattr(P['tau_v_in'], '__len__'):   	                        # For venules (inflation)
        tau_v_in  = P['tau_v_in'].astype(np.float128)             	    # Depth-specific defined by user
    else:
        tau_v_in  = (np.ones(K)*P['tau_v_in']).astype(np.float128) 	    # Default  
    if hasattr(P['tau_v_de'], '__len__'):   	                        # For venules (deflation)
        tau_v_de  = P['tau_v_de'].astype(np.float128)             	    # Depth-specific defined by user  
    else:
        tau_v_de  = (np.ones(K)*P['tau_v_de']).astype(np.float128)  	# Default  
    if hasattr(P['tau_d_in'], '__len__'):   	                        # For ascending vein (inflation)
        tau_d_in  = np.float128(P['tau_d_in']).astype(np.float128)	    # Depth-specific defined by user 
    else:
        tau_d_in  = (np.ones(K)*P['tau_d_in']).astype(np.float128)   	# Default  
    if hasattr(P['tau_d_de'], '__len__'):   	                        # For ascending vein (deflation)
        tau_d_de  = P['tau_d_de'].astype(np.float128)             	    # Depth-specific defined by user 
    else:
        tau_d_de  = (np.ones(K)*P['tau_d_de']).astype(np.float128)  	# Default
    tau_p_in      = np.float128(P['tau_p_in'])       		            # For pial vein (inflation)
    tau_p_de      = np.float128(P['tau_p_de'])       		            # For pial vein (deflation)



	##
	# Parameters for laminar BOLD signal equation (for 7 T field strenght):
	#------------------------------------------------------
	# Baseline CBV in fraction with respect to GM tissue
    V0vq = np.float128(V0v/100*K)
    V0dq = np.float128(V0d/100*K)
    V0pq = np.float128(V0p/100*K)

    TE     = np.float128(P['TE'])	 	 # echo-time (sec) 

    Hct_v  = np.float128(P['Hct_v'])	 # Hematocrit fraction
    Hct_d  = np.float128(P['Hct_d'])
    Hct_p  = np.float128(P['Hct_p'])
    B0     = np.float128(P['B0'])   	 # Field strenght        
    gyro   = np.float128(P['gyro'])      # Gyromagnetic constant 
    suscep = np.float128(P['suscep'])    # Susceptibility difference

    nu0v   = np.float128(suscep*gyro*Hct_v*B0)
    nu0d   = np.float128(suscep*gyro*Hct_d*B0)
    nu0p   = np.float128(suscep*gyro*Hct_p*B0)

	# Water proton density 
    rho_t  = np.float128(P['rho_t'])  # In GM tissue
    rho_v  = np.float128(P['rho_v'])  # In blood (venules) Ref. Lu et al. (2002) NeuroImage
    rho_d  = np.float128(P['rho_d'])  # In blood (ascening vein) 
    rho_p  = np.float128(P['rho_p'])  # In blood (pial vein) 
    rho_tp = np.float128(P['rho_tp']) # In in tissue and CSF 

	# Relaxation rates (in sec-1):
    if hasattr(P['R2s_t'],'__len__'):   	# For tissue
        R2s_t  = P['R2s_t'].astype(np.float128)
    else:
        R2s_t  = (np.ones(K)*P['R2s_t']).astype(np.float128)   	# (sec-1)
    if hasattr(P['R2s_v'],'__len__'):		# For venules
        R2s_v  = P['R2s_v'].astype(np.float128)               	# (sec-1)
    else:
        R2s_v  = (np.ones(K)*P['R2s_v']).astype(np.float128)  	# (sec-1) 
    if hasattr(P['R2s_d'],'__len__'): 		# For ascening vein
        R2s_d  = P['R2s_d'].astype(np.float128)           		# (sec-1)
    else:
        R2s_d  = (np.ones(K)*P['R2s_d']).astype(np.float128) 	# (sec-1)  
    R2s_p  = np.float128(P['R2s_p'])         			# For pial vein 

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
    dt = P['dt']
    t_steps = int(P['T']/dt)

    LBR       = np.zeros((int(P['T']/dt),K)).astype(np.float128)
    LBRpial   = np.zeros((int(P['T']/dt),K)).astype(np.float128)

    Y = {}
    Y['fa'] = np.zeros((t_steps, K))
    Y['mv'] = np.zeros((t_steps, K))
    Y['qv'] = np.zeros((t_steps, K))
    Y['qd'] = np.zeros((t_steps, K))
    Y['qp'] = np.zeros((t_steps))
	
    Y['vv'] = np.zeros((t_steps, K))
    Y['vd'] = np.zeros((t_steps, K))
    Y['vp'] = np.zeros((t_steps, K))

	##
	# Simulation
	#------------------------------------------------------
    for t in range(1, int(P['T']/dt)):

        Xk      = np.exp(Xk)    # log-normal transformation (Stephan et al.(2008), NeuroImage)
        Xp      = np.exp(Xp)
	    
	    # model input (laminar CBF response):
        f_a = (cbf[t,:].T).astype(np.float128)
	    
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

        # if t == 200:
        #     import IPython
        #     IPython.embed()

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
        Y['fa'][t,:] = f_a
        Y['mv'][t,:] = m_v
        Y['qv'][t,:] = q_v
        Y['qd'][t,:] = q_d
        Y['qp'][t]   = q_p

        Y['vv'][t,:] = v_v
        Y['vd'][t,:] = v_d
        Y['vp'][t,:] = v_p

	    
	    
        LBR[t,:] = H0*((1-V0vq-V0dq)*(k1v*V0vq*(1-q_v) +k1d*V0dq*(1-q_d)) + 
	                                    + k2v*V0vq*(1-q_v/v_v) + k2d*V0dq*(1-q_d/v_d) +
	                                    + k3v*V0vq*(1-v_v)     + k3d*V0dq*(1-v_d))*100
	    
	    
        LBRpial[t,:] = H0p*((1-V0pq)*(k1p*V0pq*(1-q_p)) + k2p*V0pq*(1-q_p/v_p) +
	                                    				  k3p*V0pq*(1-v_p))*100
        


	# save baseline physiological parameters
    Y['F0v']  = F0v
    Y['F0d']  = F0d
    Y['F0p']  = F0p

    Y['V0v']  = V0v
    Y['V0d']  = V0d
    Y['V0p']  = V0p

    Y['V0vq'] = V0vq
    Y['V0dq'] = V0dq
    Y['V0pq'] = V0pq

    Y['t0v']  = t0v
    Y['t0d']  = t0d
    Y['t0p']  = t0p
    Y['tt0v'] = tt0v
    Y['tt0d'] = tt0d
    Y['tt0']  = tt0

    return LBR, LBRpial, Y

def LBR_parameters(K, P):
	'''
	INPUT:  
		K - Number of cortical depths

	OUTPUT: 
		P - structure with all default parameters for LBR model
	
	AUTHOR: Martin Havlicek, 5 August, 2019
	'''

	#--------------------------------------------------------------------------
	# P['T'] = 30     # Default time-course duration (in seconds)

	P['K']  = K     # Number of depths

	# if K<10:
	#     P['dt'] = 0.01  # default integration step
	# elif K<20:
	#     P['dt'] = 0.005 # smaller for higher number of cortical depths
	# else:
	#     P['dt'] = 0.001


	depths = np.linspace(0,100,2*P['K']+1) # Normalized distance to the center of individual depths (in %)
	P['l']    = depths[1::2]

	# LAMINAR HEMODYNAMIC MODEL:
	#--------------------------------------------------------------------------
	# Baseline physiological parameters:
	P['V0t']   = 2.5  	# Total (regional) amount of CBV0 in the gray matter (in mL) [1-6]
	P['V0t_p'] = 1  	# Total (regional) amount of CBV0 in the pial vein (mL) [1-6]

	P['w_v'] = 0.5  # CBV0 fraction of microvasculature (i.e. venules here )with respect to the total amount 
	P['x_v'] = []   # CBV0 fraction across depths in venules 
	P['x_d'] = []   # CBV0 fraction across depths in ascending veins
	P['s_v'] = 0    # Slope of CBV increase (decrease) in venules [0-0.3]
	P['s_d'] = 0.3  # Slope of CBV increase in ascending vein     [0-1.5]

	P['t0v'] = 1    # Transit time through microvasculature(in second)
	P['E0v'] = 0.35 # Baseline oxygen extraction fraction in venules
	P['E0d'] = 0.35 # Baseline oxygen extraction fraction in ascending vein
	P['E0p'] = 0.35 # Baseline oxygen extraction fraction in pial vein

	# Parameters describing relative relationship between physiological variable:
	# CBF-CBV coupling (steady-state)
	P['alpha_v'] = 0.3 	# For venules
	P['alpha_d'] = 0.2  # For ascending vein
	P['alpha_p'] = 0.1  # For pial vein

	# CBF-CMRO2 coupling (steady-state)
	P['n'] = 4          # n-ratio   (Ref. Buxton et al. (2004) NeuroImage)

	# CBF-CBV dynamic uncoupling 
	P['tau_v_in'] = 2  # For venules - inflation 
	P['tau_v_de'] = 2  #             - deflation

	P['tau_d_in'] = 2  # For ascending vein - inflation 
	P['tau_d_de'] = 2  #                    - deflation

	P['tau_p_in'] = 2  # For pial vein - inflation 
	P['tau_p_de'] = 2  #               - deflation

	# LAMINAR BOLD SIGNAL MODEL:
	#--------------------------------------------------------------------------
	P['TE']     = 0.028     # echo-time (in sec)

	# Hematocrit fraction
	P['Hct_v']  = 0.35 	 	# For venules, Ref. Lu et al. (2002) NeuroImage
	P['Hct_d']  = 0.38		# For ascending vein
	P['Hct_p']  = 0.42  	# For pial vein


	P['B0']     = 7   					# Magnetic field strenght (in Tesla)  
	P['gyro']   = 2*np.pi*42.6*10**6  	# Gyromagnetic constant for Hydrogen
	P['suscep'] = 0.264*10**-6       	# Susceptibility difference between fully oxygenated and deoxygenated blood

	# Water proton density:
	P['rho_t']  = 0.89                 		# For gray matter tissue 
	P['rho_v']  = 0.95 - P['Hct_v']*0.22  	# For blood (venules) Ref. Lu et al. (2002) NeuroImage
	P['rho_d']  = 0.95 - P['Hct_d']*0.22  	# For blood (ascending vein)
	P['rho_p']  = 0.95 - P['Hct_p']*0.22  	# For blood (pial vein)
	P['rho_tp'] = 0.95                 		# For gray matter tissue % CSF   

	# Relaxation rates for 7 T (in sec-1)
	P['R2s_t']  = 34  # For gray matter tissue
	P['R2s_v']  = 80  # For blood (venules)
	P['R2s_d']  = 85  # For blood (ascending vein)
	P['R2s_p']  = 90  # For blood (pial vein)

	return P