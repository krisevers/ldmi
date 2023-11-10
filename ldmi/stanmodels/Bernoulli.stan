data {
  int<lower=0> N; // number of observations
  int<lower=0,upper=1> y[N]; // binary response variable
}

parameters {
  real<lower=0,upper=1> theta; // probability of success
}

model {
  // prior
  theta ~ beta(1, 1); // uniform prior
  
  // likelihood
  for (n in 1:N) {
    y[n] ~ bernoulli(theta);
  }
}
