"""
Custom Integrator Module for REBOUND
"""

import rebound
import numpy as np

# ========

default_configs = {
    'planet_num': 4,
    'planet_mass': [1e-5, 1e-5, 1e-5, 1e-5],
    'kappa': 2.000180,
    'rho': 1,
    'C': [0.5, 0.5],
}

# ========


def integrate_one_cycle(sim, target_mean_anomaly=16*np.pi, init_time_step=0.005):
    """
    Integrates to the target mean anomaly. Uses Bisection Method on sin M to find the target time. Cannot be used for circular orbit (needs to use `l` instead of `M`)

    Returns sim object
    """
    time_step = init_time_step 
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
        
        # DEBUG ONLY
        # print(N_peri, sim.t, sim.particles[1].M, current_mean_anomaly, M[-1])

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
            target_time = bisection_sin_M(sim, target_mean_anomaly, times[-2], times[-1])
            sim.integrate(target_time)
            return sim
            # return M, times, target_time


def bisection_sin_M(sim, target, a, b, tol=1e-9, doom_counts=50):
    """
    Bisection method on sin M. The function terminates after a certain attempt.
    """
    doom = 0
    while (doom <= doom_counts):
        half = (a + b)/2
        # print(a, half, b)
        
        sim.integrate(a)
        fa = np.sin(sim.particles[1].M)

        sim.integrate(b)
        fb = np.sin(sim.particles[1].M)

        sim.integrate(half)
        fhalf = np.sin(sim.particles[1].M)
        
        # If the target lies on the first half
        if fhalf > 0:
            a = a
            b = half

        # If the target lies on the second half
        if fhalf < 0:
            a = half
            b = b

        if np.abs(fa-fb) < tol:
            return half
        else:
            # Add doom counter
            doom += 1

    return half


def init_simulation(theta, configs=default_configs):
    """
    Initializes a simulation object. The system's config needs to determined in `configs`.
    """
    # Initialize the system
    planet_num = configs['planet_num']
    if planet_num <= 1:
        raise Exception("The integrator requires more than one planet.")
        
    planet_mass = configs['planet_mass']
    rho = configs['rho']
    C = configs['C']
    kappa = configs['kappa']

    # Parse theta params
    init_e = theta[:planet_num] # The first n-th elements of theta are init eccentricities
    init_M = theta[planet_num:2 * planet_num - 1] # The next (n-1)-th elements of theta are init mean anomalies
    init_pomega = theta[2 * planet_num - 1:3 * planet_num - 2] # The next (n-1)-th elements are init pomega

    # print(init_pomega)
    
    if planet_num > 2:
        init_X = theta[3 * planet_num - 2:] # Extra parameter

    # Semi-major axis defined in the paper
    sma = np.zeros(planet_num)
    sma[0] = rho * 1

    # The index 0 corresponds to the 2nd (1) planet
    period_ratio_nom = np.zeros(planet_num-1)
    period_ratio_nom[0] = kappa
    
    sma[1] = sma[0] * (period_ratio_nom[0] ** (2/3))

    for i in range(1, len(period_ratio_nom)):
        # Recursively define period_ratio_nom
        period_ratio_nom[i] = (1+C[i-1]*(1-period_ratio_nom[i-1]))**(-1)

    for i in range(2, planet_num):
        sma[i] = sma[i-1] * (init_X[i-2] + period_ratio_nom[i-1])**(2/3)
    
    # Initialize the simulation
    sim = rebound.Simulation()
    sim.add(m=1) # Primary star
    sim.add(m=planet_mass[0], a=sma[0], e=init_e[0], M=0, pomega=0) # The innermost planet

    # The other planets
    for i in range(1, planet_num):
        sim.add(m=planet_mass[i], a=sma[i], e=init_e[i], M=init_M[i-1], pomega=init_pomega[i-1])
    sim.move_to_com()    

    return sim, theta, configs, period_ratio_nom
    

def optimizing_function(theta, configs):
    """
    Calculates the square of errors (as defined in the merit function)
    """
    sim, theta, configs, period_ratio_nom = init_simulation(theta, configs)
    planet_num = configs['planet_num']

    init_e = np.array([sim.particles[i+1].e for i in range(planet_num)])
    init_M = np.array([sim.particles[i+1].M for i in range(planet_num)])
    init_pomega = np.array([sim.particles[i+1].pomega for i in range(planet_num)])


    if planet_num > 2:
        init_X = np.zeros(planet_num - 2)
        for i in range(0, len(init_X)):
            init_X[i] = sim.particles[i+3].a ** (3/2) / sim.particles[i+2].a ** (3/2) - period_ratio_nom[i] 

    # print(init_e)
    # print(init_M)
    # print(init_pomega)
    # print(init_X)
    # print()

    # Integrate
    integrate_one_cycle(sim)
    sim.move_to_com()

    # Store the final parameters. The order must be consistent with `theta`
    # Calculate diff as it goes
    final_e = np.array([sim.particles[i+1].e for i in range(planet_num)])
    final_M = np.array([sim.particles[i+1].M for i in range(planet_num)])
    final_pomega = np.array([sim.particles[i+1].pomega for i in range(planet_num)])

    if planet_num > 2:
        final_X = np.zeros(len(init_X))
        for i in range(0, len(init_X)):
            final_X[i] = sim.particles[i+3].a ** (3/2) / sim.particles[i+2].a ** (3/2) - period_ratio_nom[i] 

    e_diff = final_e - init_e
    M_diff = final_M - init_M
    X_diff = final_X - init_X

    pomega_diff = np.array([(final_pomega[i] - final_pomega[i+1]) - (init_pomega[i] - init_pomega[i+1]) for i in range(len(init_M) - 1)])
    
    # print(final_e)
    # print(final_M)
    # print(final_pomega)
    # print(final_X)
    # print()

    # print(e_diff)
    # print(M_diff)
    # print(pomega_diff)
    # print(X_diff)

    diff = np.concatenate((e_diff, M_diff[1:], pomega_diff, X_diff)).flatten()

    merit_fn = np.sum([d**2 for d in diff])

    return merit_fn


def calculate_mse(theta, configs):
    """
    Calculate mean square error of the given system after one cycle
    """
    sim, theta, configs, period_ratio_nom = init_simulation(theta, configs)
    planet_num = configs['planet_num']

    sim.move_to_com()
    cart_init = np.array([[sim.particles[i+1].x, sim.particles[i+1].y, sim.particles[i+1].z] for i in range(0, planet_num)])

    # print(cart_init)
    
    integrate_one_cycle(sim)
    sim.move_to_com()
    
    cart_final = np.array([[sim.particles[i+1].x, sim.particles[i+1].y, sim.particles[i+1].z] for i in range(0, planet_num)])

    # print(cart_final)

    cart_diff = cart_final - cart_init

    # print(cart_diff)

    mse = np.zeros(planet_num)
    for i, pos in enumerate(cart_diff):
        mse[i] = np.sum([comp ** 2 for comp in pos])

    return mse

