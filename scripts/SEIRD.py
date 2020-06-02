
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

class SIRD2:
    def __init__(self, population=44e6, beta=0.05, recovery_rate=0.033, incubation_rate=0, mortality_rate=0.036, initial_exposed=0):
        self.population = population
        self.beta = beta
        self.recovery_rate = recovery_rate
        self.incubation_rate = incubation_rate
        self.mortality_rate = mortality_rate
        self.initial_exposed = initial_exposed
    
    def _deqn(self, y, t, N, beta, gamma, delta, mu):
        E, I, R, D = y
        S = N - E - I - R - D
        dS = -beta * I * S / N
        dE = beta * I * S / N - delta * E
        dI = delta * E - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return dE, dI, dR, dD

    def simulate(self, y0, until, step=1):
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.beta,
                      self.recovery_rate,
                      self.incubation_rate,
                      self.mortality_rate,
                     )
        result = odeint(self._deqn, y0, t0, args=parameters)
        self.result = pd.DataFrame(result, columns=["E", "I", "R", "D"])
        return self.result
    
    def _opt_target(self, theta, obs, weights, estimate):
        for k, param in enumerate(estimate):
            if param in vars(self):
                vars(self)[param] = theta[k]
        y0 = np.concatenate(([self.initial_exposed], obs.head(1).to_numpy().flatten()))
        y = self.simulate(y0=y0, until=len(obs))[obs.columns]
        wsqd = weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    
    def fit(self, obs, estimate, weights=None, method="nelder-mead", options=None):
        obs = obs.reset_index()[["I", "R", "D"]]
        theta_0 = np.zeros(len(estimate))
        for k, param in enumerate(estimate):
            if param not in vars(self):
                raise Exception("Parameter not in model")
            theta_0[k] = vars(self)[param]
        if weights is None:
            weights = 1/obs.size * np.ones(obs.shape)
        args = (obs, weights, estimate)
        summary = minimize(self._opt_target, theta_0, args=args, method=method, options=options)
        return summary
