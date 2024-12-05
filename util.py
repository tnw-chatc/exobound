import rebound
import numpy as np
import argparse
import os, sys
import copy

from scipy.spatial.transform import Rotation


def absolute_distance(vec):
    abs_diff = np.zeros(len(vec))
    for i, v in enumerate(vec):
        abs_diff[i] = np.sqrt(np.sum([v[j] ** 2 for j in range(len(v))]))

    return abs_diff


def rotate_orbit(angle, params, configs):
    """
    Rotate the orbit and calculate position mean difference
    """
    planet_num = configs['planet_num']
    sim_before, _, _, _ = init_simulation(params, configs)
    sim_before.move_to_com()

    vec_before = np.array([[sim_before.particles[i+1].x, sim_before.particles[i+1].y, sim_before.particles[i+1].z] for i in range(0, planet_num)])

    sim_after, _, _, _ = init_simulation(params, configs)
    integrate_one_cycle(sim_after, configs)
    sim_after.move_to_com()
    vec_after = np.array([[sim_after.particles[i+1].x, sim_after.particles[i+1].y, sim_after.particles[i+1].z] for i in range(0, planet_num)])

    rotation = Rotation.from_euler('z', -angle) # In radians
    vec_rotated = rotation.apply(vec_after)

    vec_diff = (vec_rotated - vec_before)

    mse = np.zeros(planet_num)
    for i, pos in enumerate(vec_diff):
        mse[i] = np.sum([comp ** 2 for comp in pos])

    return np.mean(np.sqrt(mse))


def rotate_orbit_velocity(angle, params, configs):
    """
    Rotate the orbit and calculate position velocity difference
    """
    planet_num = configs['planet_num']
    sim_before, _, _, _ = init_simulation(params, configs)
    sim_before.move_to_com()

    vec_before = np.array([[sim_before.particles[i+1].vx, sim_before.particles[i+1].vy, sim_before.particles[i+1].vz] for i in range(0, planet_num)])

    sim_after, _, _, _ = init_simulation(params, configs)
    integrate_one_cycle(sim_after, configs)
    sim_after.move_to_com()
    vec_after = np.array([[sim_after.particles[i+1].vx, sim_after.particles[i+1].vy, sim_after.particles[i+1].vz] for i in range(0, planet_num)])

    rotation = Rotation.from_euler('z', -angle) # In radians
    vec_rotated = rotation.apply(vec_after)

    vec_diff = (vec_rotated - vec_before)

    mse = np.zeros(planet_num)
    for i, pos in enumerate(vec_diff):
        mse[i] = np.sum([comp ** 2 for comp in pos])

    return np.mean(np.sqrt(mse))

