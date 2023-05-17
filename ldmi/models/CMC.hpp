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


class Sim
{
	private:
		double dt;			// time step
		double PAR_sigma;	// noise amplitude
		double PAR_tau; 	// synaptic time constant
		dim2 PAR_G;			// synaptic conductance
		dim1 PAR_U;			// external input
		dim1 PAR_tau;		// time constant
		double r;			// steepness of sigmoid

		size_t nodes;
		size_t dimension;
		size_t num_steps;
		double t_sim;

		bool progress;
		
		dim1 times;
		dim3 states;
		dim2 initial_state;
		dim1 rates;
		dim1 recurrent;

	public:
		Sim(double dt,				// time step
			dim2   G,				// synaptic conductance
			dim1   U,				// external input
			dim1   tau,				// time constant
		    double t_sim,   		// simulation time
		    dim2   y,				// state vector
			bool   progress = true  // show progress bar
			) : dt(dt), PAR_G(G), PAR_U(U), PAR_tau(tau)

		{
			assert(t_sim >= 0);
			this->t_sim = t_sim;
			initial_state = y;

			nodes = y.size();
			dimension = y[0].size();
			
			num_steps = int(t_sim / dt);

			rates.resize(nodes);
			recurrent.resize(nodes);

			states.resize(num_steps);
			for (size_t i = 0; i < num_steps; ++i)
				states[i].resize(nodes);
			times.resize(num_steps);

			if (progress)
				std::cout << "Integrating " << t_sim << " seconds of simulation time with " << num_steps << " steps." << std::endl;
			this->progress = progress;
		}

		void derivative(dim2 &y, dim2 &dydt, double t)
		{
			// !TODO: work out derivate for CMC
			double dsig;
			recurrent = dot_2D1D(PAR_G, rates);
			for (int i = 0; i < nodes; i++)
			{
				// Input current
				dydt[i][0] = -y[i][0] / PAR_tau_s;
				dydt[i][0] += recurrent[i];
				dydt[i][0] += PAR_W_bg[i] * PAR_nu_bg;
				dydt[i][0] += PAR_W_ext[i] * PAR_nu_ext[i];
				dsig = sqrt(dt/PAR_tau_s) * PAR_sigma;
				dydt[i][0] += dsig * randn() / dt;

				// Membrane voltage
				dydt[i][1] = (-y[i][1] + y[i][0]*PAR_R) / PAR_tau_m;

				// Adaptation
				dydt[i][2] = (-y[i][2] + PAR_kappa[i] * y[i][3]) / PAR_tau_a;

				// Firing rate				
				rates[i] = y[i][3];
			}
		}

		double f(double x, double r)
		{
			return 1 / (1 + exp(-r * x)) - 0.5;
		}

		void integrate(std::string method)
		{
			if (method == "euler")
			{
				eulerIntegrate();
			}
			if (method == "rk4")
			{
				// rk4Integrate();
			}
		}

		void eulerIntegrate()
		{
			dim2 y = initial_state;
			double t = 0;
			for (size_t i = 0; i < num_steps; ++i)
			{
				states[i] = y;
				times[i] = t;
				euler(y, t);
				t += dt;

				if (progress)
					progress_bar(i, num_steps);
			}
		}

		void euler(dim2 &y, double t)
		{
			dim2 dydt;
			dydt.resize(nodes);
			for (int i = 0; i < nodes; ++i)
				dydt[i].resize(dimension);

			derivative(y, dydt, t);
			for (int i = 0; i < nodes; ++i)
			{
				y[i][0] += dydt[i][0] * dt;	// input current
				y[i][1] += dydt[i][1] * dt; // membrane voltage
				y[i][2] += dydt[i][2] * dt; // adaptation
				y[i][3] = f(y[i][1] - y[i][2], PAR_a, PAR_b, PAR_d); // firing rate
			}
		}

		// void rk4Integrate()
		// {
		// 	size_t n = nodes;

		// 	states[0] = initial_state;
		// 	times[0] = 0;
		// 	dim2 y = initial_state;

		// 	for (int step = 1; step < num_steps; ++step)
		// 	{
		// 		double t = step * dt;
		// 		rk4(y, t);
		// 		states[step] = y;
		// 		times[step] = 0 + t;
		// 	}
		// }

		// void rk4(dim1&y, const double t)
		// {
			
		// 	size_t n = y.size();
		// 	dim1 k1(n), k2(n), k3(n), k4(n);
		// 	dim1 f(n);
		// 	double c_dt = 1.0 / 6.0 * dt;

		// 	derivative(y, k1, t);
		// 	for (int i = 0; i < n; i++)
		// 		f[i] = y[i] + 0.5 * dt * k1[i];

		// 	derivative(f, k2, t);
		// 	for (int i = 0; i < n; i++)
		// 		f[i] = y[i] + 0.5 * dt * k2[i];

		// 	derivative(f, k3, dt);
		// 	for (int i = 0; i < n; i++)
		// 		f[i] = y[i] + dt * k3[i];

		// 	derivative(f, k4, t);
		// 	for (int i = 0; i < n; i++)
		// 		y[i] += (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]) * c_dt;
		// }

		dim3 get_states()
		{
			return states;
		}

		dim1 get_times()
		{
			return times;
		}
};
