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
		double PAR_tau_s; 	// synaptic time constant
		double PAR_tau_m;	// membrane time constant
		double PAR_C_m;		// membrane capacitance
		double PAR_R;		// membrane resistance
		dim1   PAR_kappa;	// adaptation amplitude
		double PAR_tau_a;	// adaptation time constant
		double PAR_a;
		double PAR_b;
		double PAR_d;
		double PAR_nu_bg;	// background input
		dim1   PAR_W_bg;    // background weights
		dim2   PAR_nu_ext;  // external input
		dim1   PAR_W_ext;   // external weights
		dim2   PAR_W;		// recurrent weights

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
		    double tau_s,
		    double tau_m,
			double C_m,
			dim1   kappa,
			double tau_a,
			double sigma,
			double a,
			double b,
			double d,
			double nu_bg,
			dim1   W_bg,
			dim2   nu_ext,
			dim1   W_ext,
			dim2   W,
		    double t_sim,   		// simulation time
		    dim2   y,				// state vector
			bool   progress = true  // show progress bar
			) : dt(dt), PAR_sigma(sigma), PAR_tau_s(tau_s), PAR_tau_m(tau_m), PAR_C_m(C_m), PAR_kappa(kappa), PAR_tau_a(tau_a), PAR_a(a), PAR_b(b), PAR_d(d), PAR_nu_bg(nu_bg), PAR_W_bg(W_bg), PAR_nu_ext(nu_ext), PAR_W_ext(W_ext), PAR_W(W)

		{
			assert(t_sim >= 0);
			this->t_sim = t_sim;
			initial_state = y;

			PAR_R = PAR_tau_m / PAR_C_m;

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

		void derivative(dim2 &y, dim2 &dydt, size_t it)
		{
			double dsig;
			recurrent = dot_2D1D(PAR_W, rates);
			for (int i = 0; i < nodes; i++)
			{
				// Input current
				dydt[i][0] = -y[i][0] / PAR_tau_s;
				dydt[i][0] += recurrent[i];
				dydt[i][0] += PAR_W_bg[i] * PAR_nu_bg;
				dydt[i][0] += PAR_W_ext[i] * PAR_nu_ext[i][it];
				dsig = sqrt(dt/PAR_tau_s) * PAR_sigma;
				dydt[i][0] += dsig * randn() / dt;

				// Membrane voltage
				dydt[i][1] = (-y[i][1] + y[i][0]*PAR_R) / PAR_tau_m;

				// Adaptation
				dydt[i][2] = (-y[i][2] + PAR_kappa[i] * y[i][3]) / (PAR_tau_a + 1e-10);	// 1e-10 to avoid division by zero

				// Firing rate				
				rates[i] = y[i][3];
			}
		}

		double f(double x, double a, double b, double d)
		{
			return (a * x - b) / (1 - exp(-d * (a * x - b)));
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
			for (int it = 0; it < num_steps; ++it)
			{
				states[it] = y;
				times[it] = t;
				euler(y, it);
				t += dt;

				if (progress)
					progress_bar(it, num_steps);
			}
		}

		void euler(dim2 &y, int it)
		{
			dim2 dydt;
			dydt.resize(nodes);
			for (int i = 0; i < nodes; ++i)
				dydt[i].resize(dimension);

			derivative(y, dydt, it);
			for (int i = 0; i < nodes; ++i)
			{
				y[i][0] += dydt[i][0] * dt;	// input current
				y[i][1] += dydt[i][1] * dt; // membrane voltage
				y[i][2] += dydt[i][2] * dt; // adaptation
				y[i][3] = f(y[i][1] - y[i][2], PAR_a, PAR_b, PAR_d); // firing rate
			}
		}

		dim3 get_states()
		{
			return states;
		}

		dim1 get_times()
		{
			return times;
		}
};
