import numpy as np
import pandas as pd
import rebound
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import copy
import argparse

from integrator import *

default_configs = {
        'planet_num': 4,
        'planet_mass': [1e-4, 1e-4, 1e-4, 1e-4],
        'kappa': 2.000180,
        'rho': 1,
        'C': [0.5, 0.5],
        'target_mean_anomaly': 16*np.pi,
        'init_time_step': 0.01,
        'bisection_tol': 1e-12,
    }
    
class EarlyStopping:
    def __init__(self, patience=10, save_freq=10):
        self.patience = patience
        self.current_best = np.inf
        self.save_freq = save_freq
        self.save_counter = 0
        self.count = 0

    def __call__(self, intermediate_result):
        self.save_counter += 1
        if self.save_counter % self.save_freq == self.save_freq - 1:
            print(intermediate_result.x)
        
        if intermediate_result.fun < self.current_best:
            self.current_best = intermediate_result.fun
            self.count = 0
        else:
            self.count += 1

        if self.count > self.patience:
            print("Early Stopping: Ran out of patience")
            return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patience', default=50, type=int, help="Patience")
    parser.add_argument('-n', '--n_core', default=1, type=int, help="Number of cores")
    args = vars(parser.parse_args())
    
    bounds = [(0, 0.5),(0, 0.5),(0, 0.5),(0, 0.5), 
              (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), 
              (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),
             (-.5, .5), (-.5, .5),]
    
    early_stopper = EarlyStopping(patience=args['patience'], save_freq=10)
    
    de_results = differential_evolution(optimizing_function, args=(default_configs,), bounds=bounds, 
                                        workers=args['n_core'], disp=True, callback=early_stopper,
                                       popsize=100, mutation=(0.2, 1.8), recombination=0.1)

    print(de_results)
    print()
    print(de_results.x)

