"""
Integrator Module for 3-body system ONLY.
"""

import numpy as np
import rebound
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, root, least_squares
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

import warnings


def integrate_one_cycle(sim, configs):
    time_step = configs['init_time_step']
    bisection_tol = configs['bisection_tol']
    target_mean_anomaly = configs['target_mean_anomaly']
    bisection_doom = configs['bisection_doom']
    current_mean_anomaly = 0
    N_peri = 0
    counter = 0
    M = [0]
    times = [sim.t]
    
    while True:
        # Integrate by stepping one time
        time_now = sim.t + time_step
        sim.integrate(time_now)
        
        # Getting the actual current mean anomaly
        current_mean_anomaly = sim.particles[1].M + N_peri * 2*np.pi

        # Correct mean anomaly
        if current_mean_anomaly - M[-1] < 0:
            N_peri += 1
            current_mean_anomaly = sim.particles[1].M + N_peri * 2*np.pi 
            M.append(current_mean_anomaly)
            # print(f"Starting {N_peri}")
        else:
            M.append(current_mean_anomaly)

        # Store time
        times.append(sim.t)

        # Find the precise 16pi
        if current_mean_anomaly > target_mean_anomaly:
            target_time = bisection_M(sim, target_mean_anomaly, times[-2], times[-1], M[-2], M[-1], tol=bisection_tol, doom_counts=bisection_doom)
            sim.integrate(target_time)
            return sim, target_time, times, M
            # return M, times, target_time


def bisection_M(sim, target, a, b, Ma, Mb, tol=1e-9, doom_counts=10000):
    """
    Bisection method on M. The function terminates after a certain attempt.
    """    
    func = CubicSpline([a, b], [Ma, Mb], bc_type='natural')
    count = 0

    while count < doom_counts:
        half = (a + b)/2
        
        # If the target lies on the first half
        if (func(half) - target) > 0:
            a = a
            b = half
    
        # If the target lies on the second half
        if (func(half) - target) < 0:
            a = half
            b = b
    
        # print(np.abs(fa - fb))
        
        if np.abs(a - b) < tol:
            # print(half, func(half))
            break

        count += 1

    if count >= doom_counts:
        warnings.warn("Bisection doom count reached!")
        # raise Exception("Bisection doom count reached!")

    return half

def wrap_angles(angles):
    for i, ang in enumerate(angles):
        while ang > np.pi or ang <= -np.pi:
            if ang < 0:
                ang += 2*np.pi
            elif ang > 0:
                ang -= 2*np.pi

        angles[i] = ang

    return angles

def wrap_angle(ang):
    while ang > np.pi or ang <= -np.pi:
        if ang < 0:
            ang += 2*np.pi
        elif ang > 0:
            ang -= 2*np.pi
    
    return ang


def init_simulation(theta, configs):
    inner_period = configs['inner_period']
    planet_num = configs['planet_num']
    planet_mass = configs['planet_mass']
    kappa = configs['kappa']
    C = configs['C']

    init_e = 10 ** np.array(theta[0:planet_num], dtype=np.float64)
    init_M = np.concat([[0.], np.array(theta[planet_num:2*planet_num - 1])])

    # Init pomega including the innermost planet
    init_pomega = np.zeros(planet_num)
    init_pomega[0] = 0
    for i in range(1, planet_num):
        init_pomega[i] = init_pomega[i-1] - theta[2*planet_num - 1 + (i - 1)]

    # The index 0 corresponds to the 2nd (1) planet
    period_ratio_nom = np.zeros(planet_num-1)
    period_ratio_nom[0] = kappa

    for i in range(1, planet_num - 1):
       period_ratio_nom[i] = (1+C[i-1]*(1-period_ratio_nom[i-1]))**(-1)

    period = np.zeros(planet_num)
    period[0] = inner_period
    
    for i in range(1, planet_num):
        period[i] = period[i-1] * period_ratio_nom[i-1]

    sim = rebound.Simulation()

    # Add the primary star and the innermost planet
    sim.add(m=1)

    # Add the planets
    for i in range(0, planet_num):
        sim.add(m=planet_mass[i], P=period[i], pomega=init_pomega[i], M=init_M[i], e=init_e[i])

    return sim


def optimizing_function(theta, configs):
    planet_num = configs['planet_num']
    kappa = configs['kappa']
    C = configs['C']
    
    init_theta = theta
    init_sim = init_simulation(init_theta, configs)

    final_sim, target_time, _, _ = integrate_one_cycle(init_sim, configs)
    final_sim.move_to_hel()
    
    final_e = np.log10(np.array([final_sim.particles[i+1].e for i in range(0, planet_num)]))
    final_M = wrap_angles([final_sim.particles[i+1].M for i in range(1, planet_num)])
    final_pomega = wrap_angles([final_sim.particles[i].pomega - final_sim.particles[i+1].pomega for i in range(1, planet_num)])

    # The index 0 corresponds to the 2nd (1) planet
    period_ratio_nom = np.zeros(planet_num-1)
    period_ratio_nom[0] = kappa

    for i in range(1, planet_num - 1):
       period_ratio_nom[i] = (1+C[i-1]*(1-period_ratio_nom[i-1]))**(-1)

    # Final period ratio difference (prd), formerly known as X
    if planet_num >= 3:
        final_prd = np.zeros(planet_num - 2)
        for i in range(0, planet_num - 2):
            final_prd[i] = (final_sim.particles[i+3].P / final_sim.particles[i+2].P) - period_ratio_nom[i+1]
    else:
        final_prd = []

    final_theta = np.hstack([final_e, final_M, final_pomega, final_prd])
    
    theta_diff = np.asarray(final_theta) - np.asarray(init_theta)
    # print(init_theta, final_theta)
    # print(theta_diff)

    diff = np.sum(theta_diff ** 2)
    # print(diff)
    return diff

    # return theta_diff


def calculate_mse(theta, configs, vectorize=False, verbose=False):
    """
    Calculate mean square error of the given system after one cycle
    """
    sim = init_simulation(theta, configs)
    planet_num = configs['planet_num']

    sim.move_to_hel()
    init_long = sim.particles[1].theta
    init_distance = sim.particles[1].d
    cart_init = np.array([[sim.particles[i+1].x, sim.particles[i+1].y, sim.particles[i+1].z] for i in range(0, planet_num)])

    # Apply rotation
    r = R.from_euler('z', -init_long)
    cart_init = r.apply(cart_init)
    ref_cart_init = cart_init[0]
    # print(ref_cart_init)

    integrate_one_cycle(sim, configs)
    sim.move_to_hel()

    # Transform final to be consistent with init
    final_long = sim.particles[1].theta
    final_distance = sim.particles[1].d

    long_diff = final_long - init_long
    distance_diff = final_distance - init_distance
    
    # print(init_long, final_long)
    # print(final_distance, init_distance)

    cart_final = np.array([[sim.particles[i+1].x, sim.particles[i+1].y, sim.particles[i+1].z] for i in range(0, planet_num)])
    
    # Apply rotation
    r = R.from_euler('z', -final_long)
    cart_final = r.apply(cart_final)
    ref_cart_final = cart_final[0]
    # print(ref_cart_final)

    # Apply shift
    ref_diff = ref_cart_final - ref_cart_init

    # cart_final[:,0] -= ref_diff[0]
    # cart_final[:,1] -= ref_diff[1]

    cart_diff = cart_final - cart_init

    mse = np.zeros(planet_num)
    for i, pos in enumerate(cart_diff):
        mse[i] = np.sum([comp ** 2 for comp in pos])

    if vectorize:
        return mse
    else:
        if verbose:
            print(np.sum(mse))
        return np.sum(mse)