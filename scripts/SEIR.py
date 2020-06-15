
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
    
class SIR:
    def __init__(self, population=44e6,
                 r_transmission=0.1, r_recovery=0.01, r_mortality=0.005,
                 init_infected=1, init_recovered=0, init_dead=0):
        self.population = population
        self.r_transmission = r_transmission 
        self.r_recovery = r_recovery
        self.r_mortality = r_mortality
        self.init_infected = init_infected
        self.init_recovered = init_recovered
        self.init_dead = init_dead
    
    def _deqn(self, y, t, N, beta, gamma, mu):
        I, R, D = y
        S = N - I - R - D
        #dS = -beta * S * I / N
        dI = beta * S * I / N - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return dI, dR, dD

    def simulate(self, until, step=1):
        y0 = (self.init_infected, 
              self.init_recovered, 
              self.init_dead)
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.r_transmission,
                      self.r_recovery,
                      self.r_mortality)
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
                 r_transmission=0.1, r_progression=0.05, 
                 r_recovery=0.01, r_mortality=0.005, 
                 init_exposed=0, init_infected=1, 
                 init_recovered=0, init_dead=0):
        self.population = population
        self.r_transmission = r_transmission
        self.r_progression = r_progression
        self.r_recovery = r_recovery
        self.r_mortality = r_mortality
        self.init_exposed = init_exposed
        self.init_infected = init_infected
        self.init_recovered = init_recovered
        self.init_dead = init_dead
    
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
        y0 = (self.init_exposed, 
              self.init_infected, 
              self.init_recovered, 
              self.init_dead)
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.r_transmission,
                      self.r_recovery,
                      self.r_progression,
                      self.r_mortality)
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
    def __init__(self, population=44e6, 
                 r_transmission=0.1, r_progression=0.05, 
                 r_hospitalized=0.05, r_mortality=0.005, 
                 r_recovery_mild=0.1, r_recovery_hosp=0.01,
                 init_exposed=0, init_infected=1, 
                 init_recovered=0, init_hospitalized=0, 
                 init_dead=0):
        self.population = population
        self.r_transmission = r_transmission
        self.r_progression = r_progression
        self.r_hospitalized = r_hospitalized
        self.r_recovery_mild = r_recovery_mild
        self.r_recovery_hosp = r_recovery_hosp
        self.r_mortality = r_mortality
        self.init_exposed = init_exposed
        self.init_infected = init_infected
        self.init_recovered = init_recovered
        self.init_hospitalized = init_hospitalized
        self.init_dead = init_dead
    
    def _deqn(self, y, t, N, 
              beta, gamma, delta, 
              eta, rho, mu):
        E, I, R, H, D = y
        S = N - E - I - R - H - D
        #dS = -beta * S * I
        dE = beta * S * I / N - delta * E
        dI = delta * E - (gamma + eta) * I
        dR = gamma * I + rho * H
        dH = eta * I - (mu + rho) * H
        dD = mu * H
        return dE, dI, dR, dH, dD

    def simulate(self, until, step=1):
        y0 = (self.init_exposed, 
              self.init_infected, 
              self.init_recovered,
              self.init_hospitalized,
              self.init_dead) 
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.r_transmission, 
                      self.r_recovery_mild,
                      self.r_progression,
                      self.r_hospitalized,
                      self.r_recovery_hosp,
                      self.r_mortality)
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
    def __init__(self, population=44e6,  
                 r_transmission_s=0.1, r_transmission_a=0.1, 
                 r_progression=0.05, r_asymptomatic=0.5,
                 r_hospitalized=0.05, r_mortality=0.005, 
                 r_recovery_mild=0.01, r_recovery_hosp=0.01,  
                 init_exposed=0, init_infected=1, 
                 init_recovered=0, init_hospitalized=0, 
                 init_dead=0, init_asymptomatic=0):
        self.population = population
        self.r_asymptomatic = r_asymptomatic
        self.r_transmission_s = r_transmission_s
        self.r_transmission_a = r_transmission_a
        self.r_progression = r_progression
        self.r_recovery_mild = r_recovery_mild
        self.r_hospitalized = r_hospitalized
        self.r_recovery_hosp = r_recovery_hosp
        self.r_mortality = r_mortality
        self.init_exposed = init_exposed
        self.init_infected = init_infected
        self.init_asymptomatic = init_asymptomatic
        self.init_recovered = init_recovered
        self.init_hospitalized = init_hospitalized
        self.init_dead = init_dead
    
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
        y0 = (self.init_exposed, 
              self.init_infected,
              self.init_asymptomatic,
              self.init_recovered,
              self.init_hospitalized,
              self.init_dead) 
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.r_asymptomatic,
                      self.r_transmission_s,
                      self.r_transmission_a,
                      self.r_recovery_mild,
                      self.r_progression,
                      self.r_hospitalized,
                      self.r_recovery_hosp,
                      self.r_mortality)
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
