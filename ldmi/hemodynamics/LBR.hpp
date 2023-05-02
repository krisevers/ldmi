#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fstream>
#include <iostream>

#include "utils.hpp"
#include "math.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1>   dim2;
typedef std::vector<dim2>   dim3;


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
	struct {
		double c1 = 0.6;
		double c2 = 1.5;
		double c3 = 0.6;
	} P_cbf;

	//--------------------------------------------------------------//
	// LBR: Laminar BOLD Response ----------------------------------//
	dim2 lbr;
	dim2 lbr_pial;
	int K;
	double lbr_dt;

	dim2 cmro2;

	//--------------------------------------------------------------// 
	// Laminar Hemodynamic Model -----------------------------------//
	struct {
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

		// Parameters describing relationship between elationship between physiological variables:
		// CBF-CBV coupling (steady-state)
		double alpha_v = 0.3;	// For venules
		double alpha_d = 0.2;	// For ascending vein
		double alpha_p = 0.1;	// For pial vein

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
		double gyro = 2*M_PI*42.6*10e6;	// Gyromagnetic constant for hydrogen
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
	} P;

	// Phsyiological variables
	dim1 m_v;
	dim1 v_v;
	dim1 q_v;
	dim1 v_d;
	dim1 q_d;
	dim1 v_p;
	dim1 q_p;


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

	void CerebralBloodFlow()
	{
		/* 
			Compute cerebral blood flow fron neural response
		*/
		Xvaso.resize(M);
		Yvaso.resize(M);
		Xinflow.resize(M);
		Yinflow.resize(M);
		double df_a;
		
		cbf.resize(M);
		for (int i = 0; i < M; i++)
		{
			cbf[i].resize(T);
			for (int t = 0; t < T; t++)
			{
				Xinflow[i] = exp(Xinflow[i]);
				// Vasoactive signal
				Yvaso[i] = Yvaso[i] + dt * (X[i][t] - P_cbf.c1 * Xvaso[i]);
				// Inflow
				df_a = P_cbf.c2 * Xvaso[i] - P_cbf.c3 * (Xinflow[i]-1);
				Yinflow[i] = Yinflow[i] + dt * (df_a / Xinflow[i]);

				Xvaso[i] = Yvaso[i];
				Xinflow[i] = Yinflow[i];

				cbf[i][t] = exp(Yinflow[i]);
			}
		}
	}

	void LaminarBOLDResponse()
	{
		/*
			Compute laminar BOLD response from cerebral blood flow
		*/

		// set LBR time step
		if (K < 10) 	 { lbr_dt = 0.01;  }
		else if (K < 20) { lbr_dt = 0.005; }
		else 			 { lbr_dt = 0.001; }
		int num_steps = (int)(t_sim / lbr_dt);	// number of steps in LBR

		dim1 l = linspace(100, 1, K+1); // normalized distance to the center of individual depths (in %)
		dim1 depths = std::vector<double>(l.begin(), l.end()-1);

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
		dim1 x_v = 10 + P.s_v * depths;
		dim1 x_v = x_v / sum(x_v);
		dim1 x_d = 10 + s_d * depths;
		dim1 x_d = x_d / sum(x_d);
		double w_d = 1 - P.w_v;
		dim1 V0v = P.V0t*P.w_v*x_v;	// CBV0 in venules
		dim1 V0d = P.V0t*w_d*x_d;	// CBV0 in ascending vein
		double V0p = P.V0t_p;		// CBV0 in pial vein
		// Transit time through venules (or microvasculature in general)
		dim1 t0v = ones(K) * P.t0v;
		// Depth-specific baseline CBF
		dim1 F0v = V0v / t0v;
		dim1 F0d = flipud(cumsum(flipud(F0v)));
		double F0p = F0d[0];
		// Depth-specific transit time
		dim1 t0v = V0v / F0v;
		dim1 t0d = V0d / F0d;
		double t0p = V0p / F0p;
		// (check) Total mean transit time
		double tt0v = mean(t0v);
		double tt0d = mean(cumsum(t0d));
		double tt0  = tt0v + tt0d; // must be equal to V0t/sum(F0v)
		// Baseline oxygen extraction fraction
		dim1 E0v = ones(K) * P.E0v;
		dim1 E0d = ones(K) * P.E0d;
		double E0p = P.E0p;

		// Parameters describing relative relationships between physiological variables:
		dim1 n = ones(K) * P.n; // n-ratio (= (cbf-1)/(cmro2-1)). Not used if cmro2 is directly specified as an input
		// Grubb's exponent alpha (i.e. CBF-CBV steady-state relationship)
		dim1 alpha_v = ones(K) * P.alpha_v; 	// For venules
		dim1 alpha_d = ones(K) * P.alpha_d; 	// For ascending vein
		double alpha_p = P.alpha_p; 			// For pial vein
		// CBF-CBV uncoupling (tau) during inflation and deflation
		dim1 tau_v_in = ones(K) * P.tau_v_in; // For venules 			- inflation
		dim1 tau_v_de = ones(K) * P.tau_v_de; // 			   			- deflation
		dim1 tau_d_in = ones(K) * P.tau_d_in; // For ascending vein 	- inflation
		dim1 tau_d_de = ones(K) * P.tau_d_de; // 				   		- deflation
		double tau_p_in = P.tau_p_in; 		// For pial vein 		- inflation
		double tau_p_de = P.tau_p_de; 		// 			  			- deflation

		// Parameters for laminar BOLD signal equation (for 7 T field strength):
		// Baseline CBV in fraction with respect to GM tissue
		dim1 V0vq = V0v / 100*K;
		dim1 V0dq = V0d / 100*K;
		double V0pq = V0p / 100*K;

		double TE = P.TE; // echo time (sec)

		// Hematocrit fraction
		double Hct_v = P.Hct_v; // For venules
		double Hct_d = P.Hct_d; // For ascending vein
		double Hct_p = P.Hct_p; // For pial vein
		double B0 = P.B0; 		// Magnetic field strength (Tesla)
		double gyro = P.gyro;	 	// Gyromagnetic constant
		double suscep = P.suscep;	// Susceptibility difference

		double nu0v = suscep*gyro*Hct_v*B0;
		double nu0d = suscep*gyro*Hct_d*B0;
		double nu0p = suscep*gyro*Hct_p*B0;

		// Water proton density
		double rho_t = P.rho_t; // In GM tissue
		double rho_v = P.rho_v; // In blood (venules) Ref. Lu et al. (2002) NeuroImage
		double rho_d = P.rho_d; // In blood (ascending vein)
		double rho_p = P.rho_p; // In blood (pial vein)
		double rho_tp = P.rho_tp; // In tissue and CSF
		// Relaxation rates (in sec-1)
		dim1 R2s_t = ones(K) * P.R2s_t; 	// For tissue
		dim1 R2s_v = ones(K) * P.R2s_v; 	// For blood (venules)
		dim1 R2s_d = ones(K) * P.R2s_d; 	// For blood (ascending vein)
		double R2s_p = P.R2s_p;			// For blood (pial vein)
		// (Baseline) Intra-to-extra-vascular signal ratio
		dim1 ep_v = rho_v / rho_t * exp(-TE*R2s_v) / exp(-TE*R2s_t); // For venules
		dim1 ep_d = rho_d / rho_t * exp(-TE*R2s_d) / exp(-TE*R2s_t); // For ascending vein
		double ep_p = rho_p / rho_tp * exp(-TE*R2s_p) / exp(-TE*R2s_t); // For pial vein
		// Slope of change in R2* of blood with change in extraction fraction during activation
		double r0v = 228; // For venules
		double r0d = 232; // For ascending vein
		double r0p = 226; // For pial vein

		dim1 H0 = 1 / (1 - V0vq - V0dq + ep_v * V0vq + ep_d * V0dq);
		dim1 H0p = 1 / (1 - V0pq + ep_p * V0pq);

		dim1 k1v = 4.3 * nu0v * E0v * TE;
		dim1 k2v = ep_v * r0v * E0v * TE;
		dim1 k3v = 1 - ep_v;

		dim1 k1d = 4.3 * nu0d * E0d * TE;
		dim1 k2d = ep_d * r0d * E0d * TE;
		dim1 k3d = 1 - ep_d;

		double k1p = 4.3 * nu0p * E0p * TE;
		double k2p = ep_p * r0p * E0p * TE;
		double k3p = 1 - ep_p;

		// Initial conditions
		dim1 f_a = ones(K);

		dim2 Xk = zeros(K, 4);
		dim2 Xp = zeros(1, 2);

		dim1 f_v  = ones(K);
		dim1 dv_v = ones(K);
		dim1 dHb_v = ones(K);

		dim1 f_d   = ones(K);
		dim1 dv_d  = ones(K);
		dim1 dHb_d = ones(K);

		double f_p  = 1;
		double dv_p = 1;
		double dHb_p = 1;

		dim1 tau_v = tau_v_in;
		dim1 tau_d = tau_d_in;
		double tau_p = tau_p_in;

		dim1 m = ones(K);
		
		dim2 lbr = zeros(K, num_steps);
		dim2 lbr_pial = zeros(K, num_steps);
		for (int t = 0; t < num_steps; t++)
		{
	
			// Model input (laminar CBF response)
			for (int i = 0; i < K; i++)
			{
				Xk[i][0] = exp(Xk[i][0]);	// log-normal transformation (Stephan et al. (2008), NeuroImage)
				Xk[i][1] = exp(Xk[i][1]);
				Xk[i][2] = exp(Xk[i][2]);
				Xk[i][3] = exp(Xk[i][3]);

				Xp[i][0] = exp(Xp[i][0]);
				Xp[i][1] = exp(Xp[i][1]);
				f_a[i] = cbf_sampled[i][t];
			}

			// Venules Compartments
			// --------------------------------------------
			for (int i = 0; i < K; i++)
			{
				// Blood outflow from venules compartment
				if (sum(alpha_v) > 0) {
					f_v[i] = (V0v[i] * pow(Xk[i][0], 1/alpha_v[i]) + F0v[i]*tau_v[i]*f_a[i]) / (V0v[i]+F0v[i]*tau_v[i]);
				} else {
					f_v[i] = f_a[i];
				}
				// Change in blood volume in venules
				dv_v[i] = (f_a[i] - f_v[i]) / t0v[i];
				// Change in oxygen metabolism (CMR02)
				if (cmro2.empty()) {
					m[i] = (f_a[i] + n[i]-1) / n[i];
				} else {
					m[i] = cmro2[i][t];
				}
				// Change in deoxyhemoglobin content venules
				dHb_v[i] = (m[i] - f_v[i] * Xk[i][1] / Xk[i][0]) / t0v[i];
			}

			// Ascending Vein Compartments
			// --------------------------------------------
			// Blood outflow from Kth depth of ascending vein compartment (deepest depth)
			if (alpha_d[K-1] > 0) {
				f_d[K-1] = (V0d[K-1] * pow(Xk[K-1][2], 1/alpha_d[K-1])) + tau_d[K-1] * f_v[K-1] * F0v[K-1] / (V0d[K-1] + F0d[K-1] * tau_d[K-1]);
			} else {
				f_d[K-1] = f_v[K-1] * F0v[K-1] / F0d[K-1];
			}
			// Change in blood volume and deoxyhemoglobin content in ascending vein (deepest depth)
			dv_d[K-1] = (f_v[K-1] - f_d[K-1]) / t0d[K-1];
			dHb_d[K-1] = (f_v[K-1] * Xk[K-1][1] / Xk[K-1][0] - f_d[K-1] * Xk[K-1][3] / Xk[K-1][2]) / t0d[K-1];
			// Blood outflow from other compartments of ascending vein
			for (int i = K-1; i >= 0; i--)
			{
				if (alpha_d[i] > 0) {
					f_d[i] = (V0d[i] * pow(Xk[i][2], 1/alpha_d[i]) + tau_d[i] * (f_v[i] * F0v[i] + f_d[i+1] * F0d[i+1])) / (V0d[i] + F0d[i] * tau_d[i]);
				} else {
					f_d[i] = f_v[i] * F0v[i] / F0d[i] + f_d[i+1] * F0d[i+1] / F0d[i];
				}
				// Change in blood volume and deoxyhemoglobin content in ascending vein
				dv_d[i] = (f_v[i] * F0v[i] / F0d[i] + f_d[i+1] * F0d[i+1] / F0d[i] - f_d[i]) / t0d[i];
				dHb_d[i] = (f_v[i] * F0v[i] / F0d[i] * Xk[i][1] / Xk[i][0] + f_d[i+1] * F0d[i+1] / F0d[i] * Xk[i+1][3] / X[i+1][2] - f_d[i] * Xk[i][3] / Xk[i][2]) / t0d[i];
			}

			// Pial Vein Compartments
			// --------------------------------------------
			// Blood outflow from pial vein
			if (alpha_p > 0) {
				f_p = (V0p * pow(Xp[0][1], 1/alpha_p) + F0p * tau_p * f_d[0]) / (V0p + F0p * tau_p);
			} else {
				f_p = f_d[0];
			}
			// Change in blood volume and dexoxyhemoglobin content in pial vein
			dv_p = (f_d[0] - f_p) / t0p;
			dHb_p = (f_d[0] * Xk[0][3] / Xk[0][2] - f_p * Xp[0][1] / Xp[0][0]) / t0p;


			// --------------------------------------------
			// Integrate changes to previouse time point
			for (int i = 0; i < K; i++)
			{
				Xk[i][0] += (dv_v[i] / Xk[i][0])  * dt;
				Xk[i][1] += (dHb_v[i] / Xk[i][1]) * dt;
				Xk[i][2] += (dv_d[i] / Xk[i][2])  * dt;
				Xk[i][3] += (dHb_d[i] / Xk[i][3]) * dt;

				Xp[0][0] += (dv_p / Xp[0][0])  * dt;
				Xp[0][1] += (dHb_p / Xp[0][1]) * dt;

				if (dv_v[i] < 0) {
					tau_v[i] = tau_v_de[i];
					tau_d[i] = tau_d_de[i];
					tau_p    = tau_p_de;
				}

				// venules
				m_v[i] = m[i];
				v_v[i] = exp(Xk[i][0]); // log-normal transformation
				// draining vein
				v_d[i] = exp(Xk[i][2]);
				q_d[i] = exp(Xk[i][3]);
				// pial vein
				v_p[i] = exp(Xp[0][0]);
				q_p[i] = exp(Xp[0][1]);

				// Compute BOLD Response
				lbr[i][t] = H0[i] * ((1-V0vq[i]-V0dq[i]) * (k1v[i]*V0vq[i]*(1-q_v[i]) + k1d[i]*V0dq[i]*(1-q_d[i])) + k2v[i] * V0vq[i] * (1-q_v[i]/v_v[i]) + k2d[i] * V0dq[i] * (1-q_d[i]/v_d[i]) + k3v[i] * V0vq[i] * (1-v_v[i]) + k3d[i] * V0dq[i] * (1-v_d[i])) * 100;
				lbr_pial[i][t] = H0p[i] * ((1-V0pq) * (k1p*V0pq*(1-q_p[i])) + k2p*V0pq*(1-q_p[i]/v_p[i]) + k3p*V0pq*(1-v_p[i])) * 100;
			};
		};
	};

	dim2 get_CBF()
	{
		return cbf;
	};

	dim2 get_LBR()
	{
		return lbr;
	};

	dim2 get_LBR_pial()
	{
		return lbr_pial;
	};
};