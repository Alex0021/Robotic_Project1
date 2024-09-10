
import numpy as np

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

def reynolds_input(drone_pose: np.ndarray, neighbour_poses: np.ndarray, cylinder_poses: np.ndarray, p_mig: np.ndarray=None, params: dict=dict()) -> np.ndarray:
    """
    This function calculates the command based on input poses for the Reynolds model.

    Args:
        drone_pose (np.ndarray): current drone pose [[x, y, z], [vx, vy, vz], [phi, theta, psi]]
        neighbour_poses (np.ndarray): neighbour drone poses [[x, y, z], [vx, vy, vz], [phi, theta, psi]]
        cylinder_poses (np.ndarray): obstacles poses [x, y, z] (not implemented yet)
        p_mig (np.ndarray, optional): target point to reach. Defaults to None.
        params (dict, optional): other defined parameters. Defaults to dict().

    Returns:
        np.ndarray: acceleration command
    """

    # Reynolds model parameters
    c_coh = params.get('c_coh', 1.0)
    c_sep = params.get('c_sep', 1.0)
    c_mig = params.get('c_mig', 1.0)
    c_align = params.get('c_align', 1.0)
    c_obs = params.get('c_obs', 1.0)

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

    # Compute total force
    acc = c_coh * coh - c_sep * sep + c_align * align + c_mig * migration + c_obs * obs
    acc = acc / num_neighbours

    return acc



