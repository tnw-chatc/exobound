"""
Better search for Periodic Orbit (might replace po_search.py)
"""

import rebound
import numpy as np
import argparse
import os, sys
import copy
from scipy.optimize import minimize, root

from integrator import *
from po_search import randomize_init_params


def primary_optimize(init_theta, configs, tol=1e-8, method='lm', options={}):
    """
    Primary search using Levenberg-Marquardt algorithm
    """
    planet_num = configs['planet_num']
    
    # The results is in "unbound" format. Needs to be converted
    results = root(optimizing_function_vector, x0=init_theta, args=(configs,), tol=tol, method=method, options=options)
    print(results)
    print()
    # Convert to usable format
    usable_results = copy.copy(results.x)
    usable_results[:planet_num] = sigmoid(usable_results[:planet_num])
    usable_results[-(planet_num-2):] = sigmoid_X(usable_results[-(planet_num-2):])
    print(usable_results)
    
    mse = calculate_mse(usable_results, configs)

    return usable_results, configs, mse, results.fun
    

def secondary_optimize(init_theta, configs, bounds, tol=1e-8, verbose=False, method='Nelder-Mead', options={}):
    """
    Secondary search using the customized algorithm
    """
    results = minimize(optimizing_function, *(init_theta, configs), bounds=bounds, tol=tol, method=method, options=options)
    mse = calculate_mse(results.x, configs)

    return results.x, configs, mse, results.fun


def periodic_orbit_search(init_theta, configs, bounds, cutoff=1e-1, tol_1=1e-8, tol_2=1e-8, verbose=False, method_1='lm', method_2='Nelder-Mead', options_1={}, options_2={}):
    """
    Quickly search using the primary algorithm. If it seems feasible, try the secondary algorithm which will take much longer.
    """
    print('Trying primary search...')
    results_1, configs, mse_1, _ = primary_optimize(init_theta, configs, tol_1, method=method_1, options=options_1)

    # Skip if the primary search isn't feasible
    if np.sqrt(np.mean(mse_1)) > cutoff:
        raise ValueError('Primary search is not feasible')

    print('Trying secondary search...')
    results_2, configs, mse, res_fun = secondary_optimize(results_1, configs, bounds=bounds, tol=tol_2, method=method_2, options=options_2)  
    print('Secondary search done!')

    print(results_2)

    return results_2, configs, mse, res_fun


# MAIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_trials', default=10, type=int, help="Number of trials")
    parser.add_argument('-s', '--save_path', default='results.npy', type=str, help="Saved results directory")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite the existing results")
    args = vars(parser.parse_args())
    
    n_trials = args['n_trials']

    init_param = np.zeros(n_trials, dtype='object')
    results = np.zeros(n_trials, dtype='object')

    save_path = 'results_im/'

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

    init_configs = {
        'planet_num': 4,
        'planet_mass': [1e-4, 1e-4, 1e-4, 1e-4],
        'kappa': 2.000180,
        'rho': 1,
        'C': [0.5, 0.5],
        'target_mean_anomaly': 16*np.pi,
        'init_time_step': 0.01,
        'bisection_tol': 1e-9,
    }
            
    for i in range(n_trials):
        print()
        print(f'Running... [{i+1}/{n_trials}]')
        while True:
            try:
                init_param[i] = randomize_init_params(4, [(-10, 0), (np.pi - 0.5, np.pi + 0.5), (0, 2*np.pi), (-10, 10)])
                results[i] = periodic_orbit_search(init_param[i], init_configs, bounds, tol_1=1e-6, tol_2=1e-12,
                      options_1={'ftol': 1e-3,
                               'maxiter': 10000},
                      options_2={'adaptive': True,
                              'maxiter': 10000,
                              'maxfev': 10000})
                # break if success
                break
            except ValueError as e:
                print(f'Primary search is not feasible. Retrying...')
                continue

        print(f'Done! [{i+1}/{n_trials}]')
        np.save(save_path, np.array((results[:i+1], init_param[:i+1])).transpose())
        print(f'Saved! [{i+1}/{n_trials}]')

