
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
    
class SIR:
    def __init__(self, population=44e6, transmission_rate=0.1, recovery_rate=0.01, mortality_rate=0.005,
                 initial_infected=1, initial_recovered=0, initial_dead=0):
        self.population = population
        self.transmission_rate = transmission_rate
        self.recovery_rate = recovery_rate
        self.mortality_rate = mortality_rate
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered
        self.initial_dead = initial_dead
    
    def _deqn(self, y, t, N, beta, gamma, mu):
        I, R, D = y
        S = N - I - R - D
        #dS = -beta * S * I / N
        dI = beta * S * I / N - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return dI, dR, dD

    def simulate(self, until, step=1):
        y0 = (self.initial_infected, 
              self.initial_recovered, 
              self.initial_dead)
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.transmission_rate,
                      self.recovery_rate,
                      self.mortality_rate,
                     )
        result = odeint(self._deqn, y0, t0, args=parameters)
        result_df = pd.DataFrame(result, columns=["I", "R", "D"])
        return result_df
    
    def _opt_target(self, theta, obs, weights, estimate):
        for k, param in enumerate(estimate):
            if param in vars(self):
                vars(self)[param] = theta[k]
        y = self.simulate(until=len(obs))[obs.columns]
        wsqd = weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    
    def fit(self, obs, estimate, weights=None, method="nelder-mead", options=None):
        obs = obs.reset_index(drop=True)
        self.initial_infected, self.initial_recovered, self.initial_dead = obs.head(1).to_numpy().flatten()
        theta_0 = np.zeros(len(estimate))
        for k, param in enumerate(estimate):
            if param not in vars(self):
                raise Exception("Parameter not in model")
            theta_0[k] = vars(self)[param]
        if weights is None:
            weights = 1/obs.size * np.ones(obs.shape)
        args = (obs, weights, estimate)
        result = minimize(self._opt_target, theta_0, args=args, method=method, options=options)
        return result

    
class SEIR:
    def __init__(self, population=44e6, 
                 transmission_rate=0.1, progression_rate=0.05, recovery_rate=0.01, mortality_rate=0.005, 
                 initial_exposed=0, initial_infected=1, initial_recovered=0, initial_dead=0):
        self.population = population
        self.transmission_rate = transmission_rate
        self.progression_rate = progression_rate
        self.recovery_rate = recovery_rate
        self.mortality_rate = mortality_rate
        self.initial_exposed = initial_exposed
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered
        self.initial_dead = initial_dead
    
    def _deqn(self, y, t, N, beta, gamma, delta, mu):
        E, I, R, D = y
        S = N - E - I - R - D
        #dS = -beta * I * S / N
        dE = beta * I * S / N - delta * E
        dI = delta * E - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return dE, dI, dR, dD

    def simulate(self, until, step=1):
        y0 = (self.initial_exposed, 
              self.initial_infected, 
              self.initial_recovered, 
              self.initial_dead)
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.transmission_rate,
                      self.recovery_rate,
                      self.progression_rate,
                      self.mortality_rate)
        result = odeint(self._deqn, y0, t0, args=parameters)
        self.result = pd.DataFrame(result, columns=["E", "I", "R", "D"])
        return self.result
    
    def _opt_target(self, theta, obs, weights, estimate):
        for k, param in enumerate(estimate):
            if param in vars(self):
                vars(self)[param] = theta[k]
        y = self.simulate(until=len(obs))[obs.columns]
        wsqd = weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    
    def fit(self, obs, estimate, weights=None, method="nelder-mead", options=None):
        obs = obs.reset_index(drop=True)
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

    
class SEIRH:
    def __init__(self, population=44e6, hospitalized_rate=0.03, recovery_rate_hosp=0.01,
                 transmission_rate=0.1, progression_rate=0.05, recovery_rate_mild=0.01, mortality_rate=0.005, 
                 initial_exposed=0, initial_infected=1, initial_recovered=0, initial_hospitalized=0, initial_dead=0):
        self.population = population
        self.transmission_rate = transmission_rate
        self.progression_rate = progression_rate
        self.recovery_rate_mild = recovery_rate_mild
        self.hospitalized_rate = hospitalized_rate
        self.recovery_rate_hosp = recovery_rate_hosp
        self.mortality_rate = mortality_rate
        self.initial_exposed = initial_exposed
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered
        self.initial_hospitalized = initial_hospitalized
        self.initial_dead = initial_dead
    
    def _deqn(self, y, t, N, 
              beta, gamma, delta, 
              eta, rho, mu):
        E, I, R, H, D = y
        S = N - E - I - R - H - D
        #dS = -beta * S * I
        dE = beta * S * I / N - delta * E
        dI = delta * E - (gamma + eta) * I
        dR = gamma * I + rho * H
        dH = eta * I - (rho + mu) * H
        dD = mu * H
        return dE, dI, dR, dH, dD

    def simulate(self, until, step=1):
        y0 = (self.initial_exposed, 
              self.initial_infected, 
              self.initial_recovered,
              self.initial_hospitalized,
              self.initial_dead) 
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.transmission_rate, 
                      self.recovery_rate_mild,
                      self.progression_rate,
                      self.hospitalized_rate,
                      self.recovery_rate_hosp,
                      self.mortality_rate)
        result = odeint(self._deqn, y0, t0, args=parameters)
        
        columns_names = ["E", "I", "R", "H", "D"]
        result = pd.DataFrame(result, columns=columns_names)
        return result

    def _opt_target(self, theta, obs, weights, estimate):
        for k, param in enumerate(estimate):
            if param in vars(self):
                vars(self)[param] = theta[k]
        y = self.simulate(until=len(obs))[obs.columns]
        wsqd = weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    
    def fit(self, obs, estimate, weights=None, method="nelder-mead", options=None):
        obs = obs.reset_index(drop=True)
        theta_0 = np.zeros(len(estimate))
        for k, param in enumerate(estimate):
            if param not in vars(self):
                raise Exception("Parameter not in model")
            theta_0[k] = vars(self)[param]
        if weights is None:
            weights = 1/obs.size * np.ones(obs.shape)
        args = (obs, weights, estimate)
        result = minimize(self._opt_target, theta_0, args=args, method=method, options=options)
        return result
    
class SEIARH:
    def __init__(self, population=44e6, hospitalized_rate=0.03, recovery_rate_hosp=0.01,
                 transmission_rate_s=0.1, transmission_rate_a=0.1, progression_rate=0.05, 
                 recovery_rate_mild=0.01, mortality_rate=0.005, asymptomatic_rate=0.5,
                 initial_exposed=0, initial_infected=1, initial_recovered=0, 
                 initial_hospitalized=0, initial_dead=0, initial_asymptomatic=0):
        self.population = population
        self.asymptomatic_rate = asymptomatic_rate
        self.transmission_rate_s = transmission_rate_s
        self.transmission_rate_a = transmission_rate_a
        self.progression_rate = progression_rate
        self.recovery_rate_mild = recovery_rate_mild
        self.hospitalized_rate = hospitalized_rate
        self.recovery_rate_hosp = recovery_rate_hosp
        self.mortality_rate = mortality_rate
        self.initial_exposed = initial_exposed
        self.initial_infected = initial_infected
        self.initial_asymptomatic = initial_asymptomatic
        self.initial_recovered = initial_recovered
        self.initial_hospitalized = initial_hospitalized
        self.initial_dead = initial_dead
    
    def _deqn(self, y, t, N, 
              alpha, beta_s, beta_a, gamma, delta, 
              eta, rho, mu):
        E, I, A, R, H, D = y
        S = N - E - I - A - R - H - D
        dE = beta_s * S * I / N + beta_a * S * A / N - delta * E
        dI = (1 - alpha) * delta * E - (gamma + eta) * I
        dA = alpha * delta * E - gamma * A
        dR = gamma * (I + A) + rho * H
        dH = eta * I - (rho + mu) * H
        dD = mu * H
        return dE, dI, dA, dR, dH, dD

    def simulate(self, until, step=1):
        y0 = (self.initial_exposed, 
              self.initial_infected,
              self.initial_asymptomatic,
              self.initial_recovered,
              self.initial_hospitalized,
              self.initial_dead) 
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.asymptomatic_rate,
                      self.transmission_rate_s,
                      self.transmission_rate_a,
                      self.recovery_rate_mild,
                      self.progression_rate,
                      self.hospitalized_rate,
                      self.recovery_rate_hosp,
                      self.mortality_rate)
        result = odeint(self._deqn, y0, t0, args=parameters)
        
        columns_names = ["E", "I", "A", "R", "H", "D"]
        result = pd.DataFrame(result, columns=columns_names)
        return result

    def _opt_target(self, theta, obs, weights, estimate):
        for k, param in enumerate(estimate):
            if param in vars(self):
                vars(self)[param] = theta[k]
        y = self.simulate(until=len(obs))[obs.columns]
        wsqd = weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    
    def fit(self, obs, estimate, weights=None, method="nelder-mead", options=None):
        obs = obs.reset_index(drop=True)
        theta_0 = np.zeros(len(estimate))
        for k, param in enumerate(estimate):
            if param not in vars(self):
                raise Exception("Parameter not in model")
            theta_0[k] = vars(self)[param]
        if weights is None:
            weights = 1/obs.size * np.ones(obs.shape)
        args = (obs, weights, estimate)
        result = minimize(self._opt_target, theta_0, args=args, method=method, options=options)
        return result
