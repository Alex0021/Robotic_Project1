
import numpy as np
from pyswarm_lis.obstacles import Obstacle

#===============================#
#        Reynolds Model         #
#===============================#

# This module contains the Reynolds model, which is a simple model of the
# behavior of a swarm of agents.

def get_cohesion_force(drone_pos: np.ndarray, neighbour_pos: np.ndarray, c_coh: float) -> np.ndarray:
    """
    This function calculates the cohesion force between two drones.

    Args:
        drone_pos (np.ndarray): current drone position [x, y, z]
        neighbour_pos (np.ndarray): neighbour drone position [x, y, z]
        c_coh (float): cohesion coefficient

    Returns:
        np.ndarray: cohesion force
    """
    return c_coh * (neighbour_pos - drone_pos)

def get_separation_force(drone_pos: np.ndarray, neighbour_pos: np.ndarray, c_sep: float) -> np.ndarray:
    """
    This function calculates the separation force between two drones.

    Args:
        drone_pos (np.ndarray): current drone position [x, y, z]
        neighbour_pos (np.ndarray): neighbour drone position [x, y, z]
        c_sep (float): separation coefficient

    Returns:
        np.ndarray: separation force
    """
    return c_sep * (neighbour_pos - drone_pos) / np.linalg.norm(neighbour_pos - drone_pos)**2

def get_alignment_force(drone_vel: np.ndarray, neighbour_vel: np.ndarray, c_align: float) -> np.ndarray:
    """
    This function calculates the alignment force between two drones.

    Args:
        drone_vel (np.ndarray): current drone velocity [vx, vy, vz]
        neighbour_vel (np.ndarray): neighbour drone velocity [vx, vy, vz]
        c_align (float): alignment coefficient

    Returns:
        np.ndarray: alignment force
    """
    return c_align * (neighbour_vel - drone_vel)

def get_migration_force(drone_pos: np.ndarray, p_mig: np.ndarray, c_mig: float) -> np.ndarray:
    """
    This function calculates the migration force.

    Args:
        drone_pos (np.ndarray): current drone position [x, y, z]
        p_mig (np.ndarray): target point to reach
        c_mig (float): migration coefficient

    Returns:
        np.ndarray: migration force
    """
    return c_mig * (p_mig - drone_pos)

#==============================#
#       Obstacles Handling     #
#===========================================================================
# Inspired from the paper:
# "Optimized flocking of autonomous drones in confined environments"
#===========================================================================

def compute_D_value(drone_pos: np.ndarray, cylinder_pos: np.ndarray, a: float, p: float) -> float:
    """
    This function calculates an ideal braking curve for smooth velocity control.

    Args:
        drone_pos (np.ndarray): current drone position [x, y, z]
        cylinder_pos (np.ndarray): obstacle position [x, y, z]
        a (float): preferred acceleration
        p (float): linear gain

    Returns:
        float: D value
    """
    r = np.linalg.norm(drone_pos - cylinder_pos)
    if r <= 0.0:
        return 0.0
    elif r  >= a/p:
        return np.sqrt(2*a*r - a**2/p**2)
    else:
        return r*p
    
def get_obstacle_force(drone_pos: np.ndarray, obs: "Obstacle", c_obs: float) -> np.ndarray:
    """
    This function calculates the obstacle force.
    
    ** Note: works only for cylinder obstacles. **

    Args:
        drone_pos (np.ndarray): current drone position [x, y, z]
        obs (Obstacle): obstacle definition (use Obstacle class)
        c_obs (float): obstacle coefficient

    Returns:
        np.ndarray: obstacle force
    """
    # # Obstacle parameters
    # a = 0.8
    # p = 1.0

    # u_obs = obs.center - drone_pos / np.linalg.norm(obs.center - drone_pos)
    # r_obs = obs.center - obs.radius * u_obs

    # # Compute D value
    # D = compute_D_value(drone_pos, obs.center, a, p)

    # # Compute obstacle force
    # return c_obs * (D - a) * (drone_pos - obs.center) / np.linalg.norm(drone_pos - obs.center)

    # Getting closest point on the cylinder
    u_obs = obs.center - drone_pos
    d_obs = np.linalg.norm(u_obs) - obs.radius
    u_obs = u_obs / np.linalg.norm(u_obs)

    # Compute obstacle force
    return c_obs * 1/(d_obs) * u_obs


def reynolds_input(drone_pose: np.ndarray, neighbour_poses: np.ndarray, obstacles: list["Obstacle"], p_mig: np.ndarray=None, params: dict=dict()) -> np.ndarray:
    """
    This function calculates the command based on input poses for the Reynolds model.

    Args:
        drone_pose (np.ndarray): current drone pose [[x, y, z], [vx, vy, vz], [phi, theta, psi]]
        neighbour_poses (np.ndarray): neighbour drone poses [[x, y, z], [vx, vy, vz], [phi, theta, psi]]
        obstacles (np.ndarray): obstacles definition (use Obstacle class)
        p_mig (np.ndarray, optional): target point to reach. Defaults to None.
        params (dict, optional): other defined parameters. Defaults to dict().

    Returns:
        np.ndarray: acceleration command
    """

    # Reynolds model parameters
    c_coh = params.get('c_coh', 1.0)
    c_sep = params.get('c_sep', 1.0)
    c_align = params.get('c_align', 1.0)
    c_obs = params.get('c_obs', 1.0)

    if p_mig is None:
        c_align = params.get('c_align', 1.0)
        c_mig = 0.0
    else:
        c_align = 0.0
        c_mig = params.get('c_mig', 1.0)

    drone_pos = drone_pose[0]
    drone_vel = drone_pose[1]

    num_neighbours = len(neighbour_poses)

    # loop through all the neighbours
    coh = np.zeros(3)
    sep = np.zeros(3)
    align = np.zeros(3)
    migration = np.zeros(3)
    obs = np.zeros(3)

    for neighbour_pose in neighbour_poses:
        n_pos = neighbour_pose[0]
        n_vel = neighbour_pose[1]

        # Cohesion
        coh += get_cohesion_force(drone_pos, n_pos, c_coh)
        # Separation
        sep += get_separation_force(drone_pos, n_pos, c_sep)
        # Alignment
        align += get_alignment_force(drone_vel, n_vel, c_align)
        # Migration
        migration += get_migration_force(drone_pos, p_mig, c_mig)
        # Obstacle
        for item in obstacles:
            obs += get_obstacle_force(drone_pos, item, c_obs)


    # Compute total force
    acc = coh - sep + align + migration - obs
    acc = acc / num_neighbours

    return acc



