#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fstream>
#include <iostream>

#include "utils.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1>   dim2;
typedef std::vector<dim2>   dim3;

#define PI = 3.14159265358979323846;

class Model
{
private:
	double dt;
	dim1 times;
	dim2 X;
	int M;
	int T;
	double t_sim;

	//--------------------------------------------------------------//
	// NVC: Neurovascular Coupling ---------------------------------//
	dim2 cbf;
	dim1 Xvaso;
	dim1 Yvaso;
	dim1 Xinflow;
	dim1 Yinflow;
	double c1 = 0.6;
	double c2 = 1.5;
	double c3 = 0.6;

	//--------------------------------------------------------------//
	// LBR: Laminar BOLD Response ----------------------------------//
	dim2 lbr;
	dim2 lbr_pial;
	int K;
	double lbr_dt;

	//--------------------------------------------------------------// 
	// Laminar Hemodynamic Model -----------------------------------//
	double V0t   = 2.5;	// Total (regional) amount of CBV0 in the gray matter (in mL) [1-6]
	double V0t_p = 1;	// Total (regional) amount of CBV0 in the pial vein (in mL) [1-6]

	double w_v = 0.5;	// CBV0 fraction of microvasculature (i.e. venules here) with respect to the total amount

	dim1 x_v; 			// CBV0 fraction across depths in venules
	dim1 x_d;			// CBV0 fraction across depths in ascending veins
	double s_v = 0;		// Slope of CBV increase (decrease) in venules [0-0.3]
	double s_d = 0.3;	// Slope of CBV increase in ascending vein 	   [0-1.5]

	double t0v = 1;		// Transit time through microvasculature (in second)
	double E0v = 0.35;	// Baseline oxygen extraction fraction in venules
	double E0d = 0.35;	// Baseline oxygen extraction fraction in ascending vein
	double E0p = 0.35;	// Baseline oxygen extraction fraction in pial vein

	// Parameters describing relationship between elationship between physiological variable:
	// CBF-CBV coupling (steady-state)
	double alpha_v = 0.3;		// For venules
	double alpha_d = 0.2;		// For ascending vein
	double alpha_p = 0.1;		// For pial vein

	// CBF-CMRO2 coupling (steady-state)
	double n = 4;			// n-ratio (Ref. Buxton et al. (2004) NeuroImage)

	// CBF-CBV dynamic uncoupling
	double tau_v_in = 2;	// For venules - inflation
	double tau_v_de = 2;	//			   - deflation

	double tau_d_in = 2;	// For ascending vein - inflation
	double tau_d_de = 2;	//					  - deflation

	double tau_p_in = 2;	// For pial vein - inflation
	double tau_p_de = 2;	//				 - deflation


	//--------------------------------------------------------------// 
	// Laminar BOLD Signal Model -----------------------------------//
	double TE = 0.028;		// echo-time (in sec)

	double Hct_v = 0.35; 	// For venules, Ref. Lu et al. (2002) NeuroImage
	double Hct_d = 0.38;	// For ascending vein
	double Hct_p = 0.42; 	// For pial vein

	double B0 = 7;					// Magnetic field strength (in Tesla)
	double gyro = 2*PI*42.6*10e6;	// Gyromagnetic constant for hydrogen
	double suscep = 0.264*10e-6;	// Susceptibility difference between fully oxygenated and deoxygenated blood

	// Water proton density
	double rho_t = 0.89;				// For gray matter tissue
	double rho_v = 0.95 - Hct_v*0.22;	// For blood (venules) Ref. Lu et al. (2002) NeuroImage
	double rho_d = 0.95 - Hct_d*0.22;	// For blood (ascending vein)
	double rho_p = 0.95 - Hct_p*0.22;	// For blood (pial vein)
	double rho_tp = 0.95;				// For gray matter tissue % CSF

	// Relaxation rates for 7 T (in sec-1)
	double R2s_t = 34;	// For gray matter tissue
	double R2s_v = 80;	// For blood (venules)
	double R2s_d = 85;	// For blood (ascending vein)
	double R2s_p = 90;	// For blood (pial vein)


public:
	Model(dim2 X,     // neural signal
		  double dt,  // time step
		  dim1 times  // times vector
		) : X(X), dt(dt), times(times)
	{
		this->M = X.size();					// number of nodes
		this->T = times.size();				// number of timesteps
		this->t_sim = times.size() * dt;	// simulation time
	}

	dim2 CerebralBloodFlow()
	{
		/* 
			Compute cerebral blood flow fron neural response
		*/
		Xvaso.resize(M);
		Yvaso.resize(M);
		Xinflow.resize(M);
		Yinflow.resize(M);
		double df_a;
		
		cbf.resize(M)
		for (int i = 0; i < M; i++)
		{
			cbf[i].resize(T)
			for (int t = 0; t < T; t++)
			{
				Xinflow[i] = exp(Xinflow[i]);
				// Vasoactive signal
				Yvaso[i] = Yvaso[i] + dt * (X[i][t] - c1 * Xvaso[i]);
				// Inflow
				df_a = c2 * Xvaso[i] - c3 * (Xinflow[i]-1);
				Yinflow[i] = Yinflow[i] + dt * (df_a / Xinflow[i]);

				Xvaso[i] = Yvaso[i];
				Xinflow[i] = Yinflow[i];

				cbf[i][t] = exp(Yinflow[i]);
			}
		}
	}

	dim2 LaminarBOLDResponse()
	{
		/*
			Compute laminar BOLD response from cerebral blood flow
		*/

		// set LBR time step
		if (K < 10) 	 { lbr_dt = 0.01; }
		else if (K < 20) { lbr_dt = 0.005; }
		else 			 { lbr_dt = 0.001; }
		int num_steps = (int)(t_sim / lbr_dt);	// number of steps in LBR

		dim1 l = linspace(100, 1, K+1); // normalized distance to the center of individual depths (in %)
		depths = std::vector<double>(l.begin(), l.end()-1);

		// sample num_steps from cbf
		dim2 cbf_sampled;
		cbf_sampled.resize(M);
		for (int i = 0; i < M; i++)
		{
			cbf_sampled[i].resize(num_steps);
			for (int t = 0; t < num_steps; t++)
			{
				cbf_sampled[i][t] = cbf[i][(int)(t*lbr_dt/dt)];
			}
		}
		
		// Prepare parameters and variables -----------------------------------//
		// Depth-specific CBV0
		x_v = 10 + s_v * depths;
		x_v = x_v / sum(x_v);
		x_d = 10 + s_d * depths;
		x_d = x_d / sum(x_d);
		double w_d = 1 - w_v;
		double V0v = V0t*w_v*x_v;	// CBV0 in venules
		double V0d = V0t*w_d*x_d;	// CBV0 in ascending vein
		double V0p = V0t_p;			// CBV0 in pial vein
		// Transit time through venules (or microvasculature in general)

		// Depth-specific baseline CBF
		dim1 F0v = V0v / t0v;
		dim1 F0d = 

		// Depth-specific transit time
		t0v = V0v / F0v;
		t0d = V0d / F0d;
		t0p = V0p / F0p;

		// (check) Total mean transit time
		
		// Baseline oxygen extraction fraction


		
		lbr.resize(K);
		lbr_pial.resize(K);
		for (int i = 0; i < K; i++)
		{
			lbr[i].resize(num_steps)
			lbr_pial[i].resize(num_steps);
			for (int t = 0; t < T; )
			{
				
				f_a = cbf_sampled[i][t]

				// Venules Compartments

				// Ascending Vein Compartments

				// Pial Vein Compartments
				
				lbr[i][t] = H0 * 
			}
		}
	}
}