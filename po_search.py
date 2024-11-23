"""
Modular script for search
"""

import rebound
import numpy as np
import argparse
import os, sys
from scipy.optimize import minimize

from integrator import *

# ========

def periodic_orbit_search(init_theta, configs, bounds, tol=1e-8, verbose=False):
    """
    Search for the best fitting parameter.
    """
    if verbose:
        print(f'Current Theta: {init_theta}')
        
    results = minimize(optimizing_function, *(init_theta, configs), bounds=bounds, tol=tol)
    mse = calculate_mse(results.x, configs)

    if verbose:
        print(f'Searching done!')
        print(f'mse: {mse}')
        print(f'Average mse: {np.mean(mse)}')
        print(f'Results: ')
        print(results)
        
    return results.x, configs, mse, results.fun


def randomize_init_params(planet_num, rand_bounds=None):
    """
    Generate a randomized initial parameter guess
    """
    if not rand_bounds:
        rand_bounds = [(0, 0.3), (0, 2*np.pi), (0, 2*np.pi), (-0.5, 0.5)]
        
    # Eccentricty
    e_guess = np.random.uniform(*rand_bounds[0], size=planet_num)
    M_guess = np.random.uniform(*rand_bounds[1], size=planet_num - 1)
    pomega_guess = np.random.uniform(*rand_bounds[2], size=planet_num - 1)

    if planet_num > 2:
        X_guess = np.random.uniform(*rand_bounds[3], size=planet_num - 2)
    else:
        X_guess = []

    return np.concatenate((e_guess, M_guess, pomega_guess, X_guess)).flatten()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_trials', default=10, type=int, help="Number of trials")
    parser.add_argument('-s', '--save_path', default='results.npy', type=str, help="Saved results directory")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite the existing results")
    args = vars(parser.parse_args())
    
    n_trials = args['n_trials']

    init_param = np.zeros(n_trials, dtype='object')
    results = np.zeros(n_trials, dtype='object')

    save_path = 'results/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path += args['save_path']

    if os.path.exists(save_path):
        if not args['overwrite']:
            raise FileExistsError(f'File {save_path} already exists! Use --overwrite to overwrite the existing file.')

    bounds = [(0, 0.99),(0, 0.99),(0, 0.99),(0, 0.99), 
          (None, None), (None, None), (None, None), 
          (None, None), (None, None), (None, None),
         (-.5, .5), (-.5, .5),]
            
    for i in range(n_trials):
        print()
        print(f'Running... [{i+1}/{n_trials}]')
        init_param[i] = randomize_init_params(4, [(0, 0.1), (0, np.pi), (0, np.pi), (-0.5, 0.5)])
        results[i] = periodic_orbit_search(init_param[i], default_configs, bounds, tol=1e-12, verbose=True)
        print(f'Done! [{i+1}/{n_trials}]')
        
        np.save(save_path, np.array((results[:i+1], init_param[:i+1])).transpose())
        print(f'Saved! [{i+1}/{n_trials}]')
        

