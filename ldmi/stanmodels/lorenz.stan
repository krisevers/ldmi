functions {
    vector lorenz(real t, vector y, real sigma, real rho, real beta) {
        vector[3] dydt;
        
        dydt[1] = sigma * (y[2] - y[1]);
        dydt[2] = y[1] * (rho - y[3]) - y[2];
        dydt[3] = y[1] * y[2] - beta * y[3];
        
        return dydt;
    }
}

data {
    real<lower=0> T;
    int<lower=1> N;
    real y0[3];
    real sigma;
    real rho;
    real beta;
}

transformed data {
    vector[3] y[N] = integrate_ode_rk45(lorenz, y0, 0, rep_array(T, N), sigma, rho, beta);
}

generated quantities {
    vector[3] y_gen[N] = y;
}