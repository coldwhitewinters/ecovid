
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

class CovidCalc:
    def __init__(self, 
                 population=7e6, initial_infected=1, initial_exposed=0,
                 R0=2.2, Rt=0.73, incubation_time=5.2, infectious_time=2.9, 
                 recovery_time=28.6, hospital_rate=0.2, mortality_rate=0.02,  
                 time_to_hospitalization=5, time_to_death=32, intervention_time=100):
        self.population = population
        self.R0 = R0
        self.Rt = Rt
        self.incubation_time = incubation_time
        self.infectious_time = infectious_time
        self.recovery_time = recovery_time
        self.initial_infected = initial_infected
        self.initial_exposed = initial_exposed
        self.mortality_rate = mortality_rate
        self.hospital_rate = hospital_rate
        self.time_to_hospitalization = time_to_hospitalization
        self.time_to_death = time_to_death
        self.intervention_time = intervention_time
    
    def _deqn(self, y, t, beta, t_inc, t_inf, 
              t_recovery, t_to_hosp, t_to_death,
              r_hosp, r_death):
        S, E, I, R, H, Ht, D = y
        dS = -beta * S * I
        dE = beta * S * I - 1/t_inc * E
        dI = 1/t_inc * E - 1/t_inf * I
        dR = (1 - r_hosp)/t_inf * I + 1/t_recovery * H
        dH = (r_hosp - r_death)/t_inf * I - 1/t_recovery * H
        dHt = r_death/t_inf * I - 1/(t_to_death - t_to_hosp) * Ht
        dD = 1/(t_to_death - t_to_hosp) * Ht
        return dS, dE, dI, dR, dH, dHt, dD

    def simulate(self, until, step=1):
        y0 = (self.population, self.initial_exposed, self.initial_infected, 0, 0, 0, 0) 
        t_before = np.arange(0, int(self.intervention_time), step)
        parameters = (self.R0/(self.infectious_time * self.population), 
                      self.incubation_time,
                      self.infectious_time,
                      self.recovery_time,
                      self.time_to_hospitalization,
                      self.time_to_death,
                      self.hospital_rate,
                      self.mortality_rate,
                     )
        before_intervention = odeint(self._deqn, y0, t_before, args=parameters)
        
        yt = before_intervention[-1]
        t_after = np.arange(int(self.intervention_time-1), until, step)
        parameters = (self.Rt/(self.infectious_time * self.population), 
                      self.incubation_time,
                      self.infectious_time,
                      self.recovery_time,
                      self.time_to_hospitalization,
                      self.time_to_death,
                      self.hospital_rate,
                      self.mortality_rate,
                     )
        after_intervention = odeint(self._deqn, yt, t_after, args=tuple(parameters))[1:]
        
        res = np.concatenate((before_intervention, after_intervention))
        columns_names = ["Susceptible", "Exposed", "Infectious", 
                         "Recovered", "Hosp_Moderate", "Hosp_Terminal", "Dead"]
        result = pd.DataFrame(res, columns=columns_names)
        result["Hosp_Total"] = result["Hosp_Moderate"] + result["Hosp_Terminal"]
        return result

    
class SIRD:
    def __init__(self, population=44e6, recovery_rate=0.033, mortality_rate=0.036, beta=0.5):
        self.population = population
        self.beta = beta
        self.recovery_rate = recovery_rate
        self.mortality_rate = mortality_rate
    
    def _deqn(self, y, t, N, beta, gamma, mu):
        I, R, D = y
        S = N - I - R - D
        dS = -beta * S * I / N
        dI = beta * S * I / N - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return dI, dR, dD

    def simulate(self, y0, until, step=1):
        t0 = np.arange(0, until, step)
        parameters = (self.population,
                      self.beta,
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
        y0 = obs.head(1).to_numpy().flatten()
        y = self.simulate(y0=y0, until=len(obs))
        wsqd = weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    
    def fit(self, obs, estimate, weights=None, method="nelder-mead"):
        obs = obs.reset_index()[["I", "R", "D"]]
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

    
class SEIRD:
    def __init__(self, population=44e6, beta=0.05, recovery_rate=0.033, 
                 incubation_rate=0, mortality_rate=0.036, initial_exposed=0):
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
