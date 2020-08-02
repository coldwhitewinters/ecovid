
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize

class CostQuad:
    """Weighted quadratic cost."""
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
    """Mixin with simulate and fit methods."""
    
    def simulate(self, until):
        """Simulate dynamic given by differential equation until n steps are reached."""
        y0 = tuple(self.init_values.values())
        t0 = np.arange(0, until, step=1)
        result = odeint(self._deqn, y0, t0, args=tuple(self.params.values()))
        self.y = pd.DataFrame(result, columns=self.compartment_names)
        return self.y
    
    def fit(self, obs, estimate, cost=None, method="nelder-mead", options=None):
        """Fit model parameters to observations.
        
        Parameters
        ----------
        obs : DataFrame
            Observations to fit model.
        estimate : list or tuple
            List with the names of the parameters to fit.
            Those parameters which are not in the list 
            remain unchanged.
        cost : callable
            Cost function used for fitting.
        method : str
            Optimization method used for fitting. 
            See the docs for the scipy optimization module.
        options : dict
            Parameters and configurations for the optimization method.
        """
        self.params_orig = self.params.copy()
        obs = obs.reset_index()
        
        # Set initial theta from model parameters 
        theta_0 = np.zeros(len(estimate))
        for k, key in enumerate(estimate):
            if key in self.params:
                theta_0[k] = self.params[key]
            elif key in self.init_values:
                theta_0[k] = self.init_values[key] 
            else:
                raise Exception("Wrong key")
                
        if cost is None:
            cost = CostQuad()
        
        # We need to define a function of the parameters `theta`
        # that will work as an optimization target for the optimization
        # routine. This function assigns each value of theta 
        # to the respective parameter in the model, in the order
        # established by `estimate`. Then a simulation is run and
        # the cost function is evaluated to obtain the loss 
        # associated with `theta`.
        def opt_target(theta):
            for k, key in enumerate(estimate):
                if key in self.params:
                    self.params[key] = theta[k]
                elif key in self.init_values:
                    self.init_values[key] = theta[k]
                else:
                    raise Exception("Wrong Key")
            # We make sure that we will only pick from the 
            # simulation the same columns as in the observations,
            # so we can evaulate the cost. For that we define the
            # cols variable.
            cols = obs.columns.intersection(self.compartment_names)
            y_fit = self.simulate(until=len(obs))[cols]
            return cost(y_fit, obs)

        self.fit_summary = minimize(opt_target, theta_0, method=method, options=options)
        return self.fit_summary
    
    def fit_piecewise(self, obs, estimate, cost=None, method="nelder-mead", options=None, 
                      batch_size=14, keep_remainder=False):
        """Fit model in a piecewise fashion.
        
        The idea is to divide the observations in several batches, 
        and fit a model to each batch. The initial values used for 
        fitting each batch are given by the last predicted values obtained
        after fitting to the previous batch.
        
        Parameters
        ----------
        obs : DataFrame
            Observations to fit model.
        estimate : list or tuple
            List with the names of the parameters to fit.
            Those parameters which are not in the list 
            remain unchanged.
        cost : callable
            Cost function used for fitting.
        method : str
            Optimization method used for fitting. 
            See the docs for the scipy optimization module.
        options : dict
            Parameters and configurations for the optimization method.
        batch_size : int
            Numbers of observations in each batch.
        keep_remainder : bool
            If the total number of observations is not divisible by the
            batch size, this parameter tells if we concatenate the remainder
            to the last batch or not.
        """
        self.batches = []
        self.piecewise_output = []
        self.piecewise_params = pd.DataFrame(columns=self.params.keys())

        for k in range(len(obs) // batch_size):
            self.batches.append(obs.iloc[k*batch_size:(k+1)*batch_size, :])
        
        j = len(obs) % batch_size
        if keep_remainder and j != 0:
            self.batches[-1] = pd.concat((self.batches[-1], obs.iloc[-j:, :]))
    
        for k in range(0, len(self.batches)):
            print("----------------------")
            print("Batch", k)
            print()

            print("Fitting...")
            self.fit(
                self.batches[k], 
                estimate=estimate,
                cost=cost,
                method=method,
                options=options,
            ) 

            print("Simulating...")
            res = self.simulate(until=len(self.batches[k]) + 1)
            res, y0 = res.iloc[:-1, :], res.iloc[-1, :]
            res.index = self.batches[k].index
            
            for k, c in zip(self.init_values, self.compartment_names):
                self.init_values[k] = y0[c]

            self.piecewise_output.append(res)
            self.piecewise_params = self.piecewise_params.append(self.params, ignore_index=True)
            print("Done!")
        
        self.y_fit = pd.concat(self.piecewise_output)
    

class SIR(FitSimulateMixin):
    """SIR Model.
    
    Parameters
    ----------
    population : float
        Susceptible population.
    r_transmission : float
        Rate of transmission.
    r_recovery : float
        Rate of recovery.
    r_mortality : float
        Rate of mortality.
    init_infected : float
        Initial infected.
    init_recovered : float
        Initial recovered.
    init_dead : float
        Initial dead.
    """
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
    """SEIR Model.
    
    Parameters
    ----------
    population : float
        Susceptible population.
    r_transmission : float
        Rate of transmission.
    r_progression : float
        Rate of progression from exposed to infected
    r_recovery : float
        Rate of recovery.
    r_mortality : float
        Rate of mortality.
    init_exposed : float
        Initial exposed.
    init_infected : float
        Initial infected.
    init_recovered : float
        Initial recovered.
    init_dead : float
        Initial dead.
    """    
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
    """SEIRH Model.
    
    Parameters
    ----------
    population : float
        Susceptible population.
    r_transmission : float
        Rate of transmission.
    r_progression : float
        Rate of progression from exposed to infected.
    r_hospitalized : float
        Rate of hospitalization.
    r_recovery_mild : float
        Rate of recovery for cases not requiring hospitalization.
    r_recovery_hosp : float
        Rate of recovery for cases requiring hospitalization.
    r_mortality : float
        Rate of mortality.
    init_exposed : float
        Initial exposed.
    init_infected : float
        Initial infected.
    init_recovered : float
        Initial recovered.
    init_hospitalized : float
        Initial hospitalized.
    init_dead : float
        Initial dead.
    """    
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
