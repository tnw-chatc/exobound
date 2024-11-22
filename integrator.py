"""
Custom Integrator Module for REBOUND
"""

import rebound
import numpy as np


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
    Bisection method on sin M. The function terminates after the 50th unsuccessful attempt.
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


