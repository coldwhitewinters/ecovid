
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize

class CostQuad:
    def __init__(self, weights=None):
        self.weights = weights
    
    def __call__(self, y, obs):
        if self.weights is None:
            wsqd = (y - obs)**2
        else:
            wsqd = self.weights * (y - obs)**2
        cost = wsqd.sum().sum()
        return cost
    

class FitSimulateMixin:
    def simulate(self, until, step=1):
        y0 = tuple(self.init_values.values())
        t0 = np.arange(0, until, step)
        result = odeint(self._deqn, y0, t0, args=tuple(self.params.values()))
        self.output = pd.DataFrame(result, columns=self.compartment_names)
        return self.output
    
    def fit(self, obs, estimate, cost=None, method="nelder-mead", options=None):
        self.params_orig = self.params.copy()
        obs = obs.reset_index(drop=True)
        theta_0 = np.array([self.params[key] for key in estimate])
        if cost is None:
            cost = CostQuad()
        
        def opt_target(theta):
            for k, key in enumerate(estimate):
                self.params[key] = theta[k]
            y = self.simulate(until=len(obs))[obs.columns]
            return cost(y, obs)

        self.fit_summary = minimize(opt_target, theta_0, method=method, options=options)
        return self.fit_summary


class SIR(FitSimulateMixin):
    def __init__(self, population=44e6,
                 r_transmission=0.1, r_recovery=0.01, r_mortality=0.005,
                 init_infected=1, init_recovered=0, init_dead=0):
        self.init_values = {
            "init_infected": init_infected,
            "init_recovered": init_recovered,
            "init_dead": init_dead,
        }
        self.params = {
            "population": population,
            "r_transmission": r_transmission,
            "r_recovery": r_recovery,
            "r_mortality": r_mortality,
        }
        self.compartment_names = ["I", "R", "D"]
    
    def _deqn(self, y, t, N, beta, gamma, mu):
        I, R, D = y
        S = N - I - R - D
        #dS = -beta * S * I / N
        dI = beta * S * I / N - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return dI, dR, dD
    
    
class SEIR(FitSimulateMixin):
    def __init__(self, population=44e6,
                 r_transmission=0.1, r_progression=0.05, 
                 r_recovery=0.01, r_mortality=0.005, 
                 init_exposed=0, init_infected=1, 
                 init_recovered=0, init_dead=0):
        self.params = {
            "population": population,
            "r_transmission": r_transmission,
            "r_progression": r_progression,
            "r_recovery": r_recovery,
            "r_mortality": r_mortality,
        }
        self.init_values = {
            "init_exposed": init_exposed,
            "init_infected": init_infected,
            "init_recovered": init_recovered,
            "init_dead": init_dead,
        }
        self.compartment_names = ["E", "I", "R", "D"]
    
    def _deqn(self, y, t, N, beta, gamma, delta, mu):
        E, I, R, D = y
        S = N - E - I - R - D
        #dS = -beta * I * S / N
        dE = beta * I * S / N - delta * E
        dI = delta * E - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return dE, dI, dR, dD

    
class SEIRH(FitSimulateMixin):
    def __init__(self, population=44e6, 
                 r_transmission=0.1, r_progression=0.05, 
                 r_hospitalized=0.05, r_mortality=0.005, 
                 r_recovery_mild=0.1, r_recovery_hosp=0.01,
                 init_exposed=0, init_infected=1, 
                 init_recovered=0, init_hospitalized=0, 
                 init_dead=0):
        self.params = {
            "population": population,
            "r_transmission": r_transmission,
            "r_progression": r_progression,
            "r_hospitalized": r_hospitalized,
            "r_recovery_mild": r_recovery_mild,
            "r_recovery_hosp": r_recovery_hosp,
            "r_mortality": r_mortality,
        }
        self.init_values = {
            "init_exposed": init_exposed,
            "init_infected": init_infected,
            "init_recovered": init_recovered,
            "init_hospitalized": init_hospitalized,
            "init_dead": init_dead,
        }
        self.compartment_names = ["E", "I", "R", "H", "D"]
    
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
