#include <vector>
#include <string>
#include <assert.h>
#include <iostream>

#include "utils.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1>   dim2;

class Sim
{
	private:
		double dt;
		double PAR_rho;
		double PAR_sigma;
		double PAR_beta;

		size_t dimension;
		size_t num_steps;
		double t_sim;

		bool progress;

		dim1 times;
		dim2 states;
		dim1 initial_state;

	public:
		Sim(double dt,		// time step
		   double rho,
		   double sigma,
		   double beta,
		   double t_sim,   	// simulation time
		   dim1   y,			// state vector
		   bool progress = true  // show progress bar
		   ) : dt(dt), PAR_rho(rho), PAR_sigma(sigma), PAR_beta(beta)
		{
			assert(t_sim >= 0);
			this->t_sim = t_sim;
			initial_state = y;

			dimension = y.size();
			num_steps = int(t_sim / dt);

			states.resize(num_steps);
			for (size_t i = 0; i < num_steps; ++i)
				states[i].resize(dimension);
			times.resize(num_steps);

			if (progress == true)
				std::cout << "Integrating " << t_sim << " seconds of "
				          << dimension << "-dimensional Lorenz system "
				          << "with dt = " << dt << " seconds." << std::endl;
			this->progress = progress;
		}

		void derivative(dim1 &y, dim1 &dydt, double t)
		{
			dydt[0] = PAR_sigma * (y[1] - y[0]);
			dydt[1] = y[0] * (PAR_rho - y[2]) - y[1];
			dydt[2] = y[0] * y[1] - PAR_beta * y[2];
		}

		void integrate(std::string method)
		{
			if (method == "euler")
			{
				eulerIntegrate();
			}
			if (method == "rk4")
			{
				rk4Integrate();
			}
		}

		void eulerIntegrate()
		{
			dim1 y = initial_state;
			double t = 0;
			for (size_t i = 0; i < num_steps; ++i)
			{
				states[i] = y;
				times[i] = t;
				euler(y, t);
				t += dt;
				if (progress == true)
					progress_bar(i, num_steps);
			}
		}

		void euler(dim1 &y, double t)
		{
			dim1 dydt(dimension);
			derivative(y, dydt, t);
			for (size_t i = 0; i < dimension; ++i)
				y[i] += dydt[i] * dt;
		}

		void rk4Integrate()
		{
			size_t n = dimension;

			states[0] = initial_state;
			times[0] = 0;
			dim1 y = initial_state;

			for (int step = 1; step < num_steps; ++step)
			{
				double t = step * dt;
				rk4(y, t);
				states[step] = y;
				times[step] = 0 + t;
				if (progress == true)
					progress_bar(step, num_steps);
			}
		}

		void rk4(dim1&y, const double t)
		{
			
			size_t n = y.size();
			dim1 k1(n), k2(n), k3(n), k4(n);
			dim1 f(n);
			double c_dt = 1.0 / 6.0 * dt;

			derivative(y, k1, t);
			for (int i = 0; i < n; i++)
				f[i] = y[i] + 0.5 * dt * k1[i];

			derivative(f, k2, t);
			for (int i = 0; i < n; i++)
				f[i] = y[i] + 0.5 * dt * k2[i];

			derivative(f, k3, dt);
			for (int i = 0; i < n; i++)
				f[i] = y[i] + dt * k3[i];

			derivative(f, k4, t);
			for (int i = 0; i < n; i++)
				y[i] += (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]) * c_dt;
		}

		dim2 get_states()
		{
			return states;
		}

		dim1 get_times()
		{
			return times;
		}
};
