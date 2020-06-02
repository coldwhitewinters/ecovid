
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

class SIRD:
    def __init__(self, population=44e6, recovery_rate=0.033, mortality_rate=0.036, alpha=0.05, beta=0.05):
        self.population = population
        self.alpha = alpha
        self.beta = beta
        self.recovery_rate = recovery_rate
        self.mortality_rate = mortality_rate
    
    def _deqn(self, y, t, N, alpha, beta, gamma, mu):
        Ia, Is, R, D = y
        S = N - Ia - Is - R - D
        dS = (-alpha * Ia - beta * Is) * S / N
        dIa = alpha * S * Ia / N - gamma * Ia 
        dIs = beta * S * Is / N - (gamma + mu) * Is
        dR = gamma * (Ia + Is)
        dD = mu * Is
        return dIa, dIs, dR, dD

    def simulate(self, y0, until, step=1):
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.alpha,
                      self.beta,
                      self.recovery_rate,
                      self.mortality_rate,
                     )
        result = odeint(self._deqn, y0, t0, args=parameters)
        result_df = pd.DataFrame(result, columns=["Ia", "Is", "R", "D"])
        return result_df
    
    def _opt_target(self, theta, obs, weights, estimate):
        for k, param in enumerate(estimate):
            if param in vars(self):
                vars(self)[param] = theta[k]
        y0 = obs.head(1).to_numpy().flatten()
        y = self.simulate(y0=y0, until=len(obs))
        wsqd = weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    
    def fit(self, obs, estimate, weights=None, method="nelder-mead"):
        obs = obs.reset_index()[["Is", "R", "D"]]
        theta_0 = np.zeros(len(estimate))
        for k, param in enumerate(estimate):
            if param not in vars(self):
                raise Exception("Parameter not in model")
            theta_0[k] = vars(self)[param]
        if weights is None:
            weights = 1/obs.size * np.ones(obs.shape)
        args = (obs, weights, estimate)
        result = minimize(self._opt_target, theta_0, args=args, method=method)
        return result
