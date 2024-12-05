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
    while True:
        print('Trying primary search...')
        results_1, configs, mse_1, _ = primary_optimize(init_theta, configs, tol_1, method=method_1, options=options_1)

        # Skip if the primary search isn't feasible
        if np.sqrt(np.mean(mse_1)) > cutoff:
            print('Primary search is not feasible... Skipping...')
            continue

        print('Trying secondary search...')
        results_2, configs, mse, res_fun = secondary_optimize(results_1, configs, bounds=bounds, tol=tol_2, method=method_2, options=options_2)  
        print('Secondary search done!')

        print(results_2)

        return results_2, configs, mse, res_fun

