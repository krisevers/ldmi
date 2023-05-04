// STAN implementation of Lorenz system
// https://en.wikipedia.org/wiki/Lorenz_system

data {
    int < lower = 1 > N; // Sample size
    int < lower = 1 > T; // Number of time steps
    real dt;             // Time step
    real x[N, T]; // X position
    real y[N, T]; // Y position
    real z[N, T]; // Z position
}

parameters {
    real<lower=0, upper=20> s;  // 
    real<lower=0, upper=10> b;  // 
    real<lower=0, upper=40> r;  //
}

model {
    s ~ uniform(10, 10);
    b ~ uniform(8/3, 10);
    r ~ uniform(28, 10);
}

functions {
    real[] f(real N, real T, real dt, real[] theta) {
        real s = theta[1]; // sigma
        real b = theta[2]; // beta
        real r = theta[3]; // rho
        // states
        real x[N, T];
        real y[N, T];
        real z[N, T];
        // initial conditions
        x[:, 1] = 0;
        y[:, 1] = 1;
        z[:, 1] = 1.05;
        // time loop
        for (t in 2:T) {
            for (n in 1:N) {
                x[n, t] = (s * (y[n, t-1] - x[n, t-1]))             * dt + x[n, t-1];
                y[n, t] = (x[n, t-1] * (r - z[n, t-1]) - y[n, t-1]) * dt + y[n, t-1];
                z[n, t] = (x[n, t-1] * y[n, t-1] - b * z[n, t-1])   * dt + z[n, t-1];
            }
        }
        return {x, y, z};
    }
}