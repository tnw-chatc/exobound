import numpy as np
import rebound
from scipy.spatial.transform import Rotation as R

# default_configs = {
#     'planet_num': 4,
#     'planet_mass': [1e-5, 1e-5, 1e-5, 1e-5],
#     'kappa': 2.000180,
#     'rho': 1,
#     'C': [0.5, 0.5],
#     'target_mean_anomaly': 16*np.pi,
#     'init_time_step': 0.02,
#     'bisection_tol': 1e-9,
# }


def integrate_one_cycle(sim, configs):
    """
    Integrates to the target mean anomaly. Uses Bisection Method on sin M to find the target time. Cannot be used for circular orbit (needs to use `l` instead of `M`)

    Returns sim object
    """
    time_step = configs['init_time_step']
    target_mean_anomaly = configs['target_mean_anomaly']
    bisection_tol = configs['bisection_tol']
    current_mean_anomaly = 0
    N_peri = 0
    counter = 0
    M = [0]
    times = [sim.t]

    # If target time override is parsed, return the integrated simulation object with such time
    try:
        overridden_target_time = configs['overridden_target_time']
    except KeyError as e:
        overridden_target_time = None

    if overridden_target_time:
        target_time = overridden_target_time
        sim.integrate(overridden_target_time)
        return sim, target_time
    
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
            target_time = bisection_sin_M(sim, target_mean_anomaly, times[-2], times[-1], bisection_tol)
            sim.integrate(target_time)
            return sim, target_time
            # return M, times, target_time


def bisection_sin_M(sim, target, a, b, tol=1e-9, doom_counts=100):
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


def wrap_angle(angles):
    for i, ang in enumerate(angles):
        while ang > np.pi or ang <= -np.pi:
            if ang < 0:
                ang += 2*np.pi
            elif ang > 0:
                ang -= 2*np.pi

        angles[i] = ang

    return angles


def init_simulation(theta, configs):
    """
    Initializes a simulation object. The system's config needs to determined in `configs`.

    Focuses on period rather than semi-major axis.

    Reworked for 3 planets or more systems.
    """
    planet_num = configs['planet_num']
    planet_mass = configs['planet_mass']
    kappa = configs['kappa']
    C = configs['C']

    if planet_num <= 2:
        raise Exception("The number of planets must be 3 or greater.")

    init_e = 10 ** np.array(theta[0:planet_num], dtype=np.float64)
    init_M = np.array(theta[planet_num:2*planet_num - 1])
    init_pomega = np.array(theta[2*planet_num - 1:3*planet_num - 2])
    init_X = np.array(theta[3*planet_num - 2:])

    # The index 0 corresponds to the 2nd (1) planet
    period_ratio_nom = np.zeros(planet_num-1)
    period_ratio_nom[0] = kappa

    for i in range(1, planet_num - 1):
       period_ratio_nom[i] = (1+C[i-1]*(1-period_ratio_nom[i-1]))**(-1)

    period = np.zeros(planet_num)
    period[0] = 2*np.pi
    
    for i in range(1, planet_num):
        period[i] = period[i-1] * period_ratio_nom[i-1]

    # Initialize the simulation
    sim = rebound.Simulation()

    if configs['normalize_mass']:
        star_mass = 1 - np.sum(planet_mass)
    else:
        star_mass = 1
        
    sim.add(m=star_mass) # Primary star
    sim.add(m=planet_mass[0], P=period[0], pomega=0, M=0, e=init_e[0])
    for i in range(1, planet_num):
        sim.add(m=planet_mass[i], P=period[i], pomega=init_pomega[i-1], M=init_M[i-1], e=init_e[i])

    sim.move_to_com()
    
    return sim, theta, configs


def optimizing_function(theta, configs, vectorize=False):
    """
    Calculates the square of errors (as defined in the merit function)
    """
    sim, theta, configs = init_simulation(theta, configs)
    sim.move_to_hel()
    planet_num = configs['planet_num']
    kappa = configs['kappa']
    C = configs['C']

    init_theta = np.zeros((planet_num, 4))
    init_long = sim.particles[planet_num].theta
    for i in range(planet_num):
        x = sim.particles[i+1].x
        y = sim.particles[i+1].y
        vx = sim.particles[i+1].vx
        vy = sim.particles[i+1].vx
        init_theta[i, :] = [x, y, vx, vy]
        
    # Integrate
    integrate_one_cycle(sim, configs)
    sim.move_to_hel()

    final_theta = np.zeros((planet_num, 4))
    final_long = sim.particles[planet_num].theta

    long_diff = final_long - init_long
    
    for i in range(planet_num):
        # Transform final theta to be consistent with init theta
        x = sim.particles[i+1].x
        y = sim.particles[i+1].y
        pos_vec = np.array([x, y, 0])
        r = R.from_euler('z', -long_diff)
        pos_vec_transformed = r.apply(pos_vec)
        
        vx = sim.particles[i+1].vx
        vy = sim.particles[i+1].vx
        vel_vec = np.array([vx, vy, 0])
        r = R.from_euler('z', -long_diff)
        vel_vec_transformed = r.apply(vel_vec)
        
        cart_vec = np.array([pos_vec_transformed[:2], vel_vec_transformed[:2]]).flatten()
        
        final_theta[i, :] = cart_vec

    diff = final_theta - init_theta

    if vectorize:
        return diff
    else:
        return np.sum(diff ** 2)


def optimizing_function_keplerian(theta, configs, vectorize=False):
    """
    Calculates the square of errors (as defined in the merit function)
    """
    sim, theta, configs = init_simulation(theta, configs)
    planet_num = configs['planet_num']
    kappa = configs['kappa']
    C = configs['C']
    sim.move_to_hel()

    # The index 0 corresponds to the 2nd (1) planet
    period_ratio_nom = np.zeros(planet_num-1)
    period_ratio_nom[0] = kappa

    for i in range(1, planet_num - 1):
       period_ratio_nom[i] = (1+C[i-1]*(1-period_ratio_nom[i-1]))**(-1)

    # Initial Keplerian parameters
    init_e = np.log(np.array([sim.particles[i+1].e for i in range(planet_num)]))
    init_M = np.array([sim.particles[i+1].M for i in range(planet_num)])
    init_pomega = np.array([sim.particles[i+1].pomega for i in range(planet_num)])

    init_X = np.zeros(planet_num - 2)
    for i in range(0, len(init_X)):
        init_X[i] = sim.particles[i+3].P / sim.particles[i+2].P - period_ratio_nom[i+1] 

    # Integrate
    integrate_one_cycle(sim, configs)
    sim.move_to_hel()

    # Final Keplerian parameters
    final_e = np.log(np.array([sim.particles[i+1].e for i in range(planet_num)]))
    final_M = np.array([sim.particles[i+1].M for i in range(planet_num)])
    final_pomega = np.array([sim.particles[i+1].pomega for i in range(planet_num)])

    final_X = np.zeros(planet_num - 2)
    for i in range(0, len(init_X)):
        final_X[i] = sim.particles[i+3].P / sim.particles[i+2].P - period_ratio_nom[i+1] 

    e_diff = final_e - init_e
    M_diff = wrap_angle(final_M) - wrap_angle(init_M)
    X_diff = final_X - init_X

    pomega_diff = np.array([(final_pomega[i] - final_pomega[i+1]) - (init_pomega[i] - init_pomega[i+1]) for i in range(len(init_M) - 1)])

    diff = np.concatenate((e_diff, M_diff[1:], pomega_diff, X_diff)).flatten()
    diff = np.concatenate((e_diff, M_diff[1:], pomega_diff)).flatten()

    if vectorize:
        return diff
    else:
        return np.sum(diff ** 2)


def calculate_mse(theta, configs, vectorize=False, verbose=False):
    """
    Calculate mean square error of the given system after one cycle
    """
    sim, theta, configs = init_simulation(theta, configs)
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


