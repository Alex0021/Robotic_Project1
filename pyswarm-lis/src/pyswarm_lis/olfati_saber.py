import numpy as np
from pyswarm_lis.obstacles import Obstacle, Cylinder

'''
    This code is greatly inspired from the swarm_pilot.py file
    coded by Benjamin Jarvis for his doctoral thesis.
    All rights reserved
'''

#===============================#
#    Frame transformations      #
#===============================#

def get_RB2W(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Get the rotation matrix from the body frame to the world frame

    Args:
        phi (float): angle around the x-axis (roll)
        theta (float):  angle around the y-axis (pitch)
        psi (float): angle around the z-axis (yaw)

    Returns:
        np.ndarray: rotation matrix from the body frame to the world frame
    """
    R_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0,0,1]])
    R_theta = np.array([[np.cos(theta), 0,np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]])
    R_phi = [[1,0,0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    return R_psi @ R_theta @ R_phi

def get_W2B(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Get the rotation matrix from the world frame to the body frame

    Args:
        phi (float): angle around the x-axis (roll)
        theta (float):  angle around the y-axis (pitch)
        psi (float): angle around the z-axis (yaw)

    Returns:
        np.ndarray: rotation matrix from the world frame to the body frame
    """
    R_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0,0,1]])
    R_theta = np.array([[np.cos(theta), 0,np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]])
    R_phi = [[1,0,0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    return np.transpose(R_phi @ R_theta @ R_psi)

def rot_global2body(input: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Rotate a vector from the global frame to the body frame

    Args:
        input (np.ndarray): vector to rotate
        angles (np.ndarray): angles of the body frame [phi, theta, psi]

    Returns:
        np.ndarray: rotated vector
    """
    R = get_W2B(angles[0], angles[1], angles[2])
    return R @ input

def rot_body2global(input: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Rotate a vector from the body frame to the global frame

    Args:
        input (np.ndarray): vector to rotate
        angles (np.ndarray): angles of the body frame [phi, theta, psi]

    Returns:
        np.ndarray: rotated vector
    """ 
    R = get_RB2W(angles[0], angles[1], angles[2])
    return R @ input

#===============================#
#    Olfati-Saber Model        #
#===============================#

# Calculate the cohesion intensity for the Olfati-Saber model
def get_cohesion_intensity(r: float, d_ref: float, a: float, b: float, c: float) -> float:
    """
    Calculate the cohesion intensity for the Olfati-Saber model

    Args:
        r (float): distance between two drones
        d_ref (float): desired distance between two drones
        a (float): potential field parameter
        b (float): potential field parameter
        c (float): potential field parameter

    Returns:
        float: cohesion force
    """
    
    diff = r - d_ref
    return ((a+b)/2 * (np.sqrt(1+(diff + c)**2) - np.sqrt(1+c**2)) + (a-b)*diff/2)

# Calculate the cohesion intensity derivative for the Olfati-Saber model
def get_cohesion_intensity_der(r: float, d_ref: float, a: float, b: float, c: float) -> float:
    """
    Calculate the cohesion intensity derivative for the Olfati-Saber model

    Args:
        r (float): distance between two drones
        d_ref (float): desired distance between two drones
        a (float): potential field parameter
        b (float): potential field parameter
        c (float): potential field parameter

    Returns:
        float: cohesion force derivative
    """
        
    diff = r - d_ref
    return (a+b)/2 * (diff + c) / np.sqrt(1+(diff + c)**2) + (a-b)/2

# Calcualte the neighbour weight for the Olfati-Saber model
def get_neighbour_weight(r: float|np.ndarray, r0: float, delta: float) -> float:
    """
    Calcualte the neighbour weight for the Olfati-Saber model

    Args:
        r (float): distance between current drone and neighbour drone
        r0 (float): minimum distance to be part of the swarm
        delta (float): parameter for the neighbour weight

    Returns:
        float: neighbour weight
    """
    r_ratio = r / r0

    if isinstance(r, float):
        if r_ratio < delta:
            return 1
        elif r_ratio < 1:
            return 0.25 * (1 + np.cos(np.pi * (r_ratio - delta) / (1 - delta)))**2
        else:
            return 0
    elif isinstance(r, np.ndarray):
        nb_elems = len(r)
        output = np.zeros(nb_elems)

        for i in range(nb_elems):
            if r_ratio[i] < delta:
                output[i] = 1
            elif r_ratio[i] < 1:
                output[i] = 0.25 * (1 + np.cos(np.pi * (r_ratio[i] - delta) / (1 - delta)))**2

        return output

# Calcualte the derivative of the neighbour weight for the Olfati-Saber model
def get_neighbour_weight_der(r: float, r0: float, delta: float) -> float:
    """
    Calcualte the derivative of the neighbour weight for the Olfati-Saber model

    Args:
        r (float): distance between current drone and neighbour drone
        r0 (float): minimum distance to be part of the swarm
        delta (float): parameter for the neighbour weight

    Returns:
        float: neighbour weight derivate
    """
    r_ratio = r/r0

    if r_ratio < delta:
        return 0
    elif r_ratio < 1:
        arg = np.pi * (r_ratio - delta) / (1 - delta)
        return 1/2*(-np.pi/(1-delta))*(1+np.cos(arg))*np.sin(arg)
    else:
        return 0

# Calculate the attraction/repulsion force for the Olfati-Saber model
def get_cohesion_force(r: float, d_ref: float, a: float, b: float, c: float, r0: float, delta: float) -> float:
    """
    Calculate the attraction/repulsion force for the Olfati-Saber model

    Args:
        r (float): distance between current drone and neighbour drone
        d_ref (float): desired distance between members of the swarm
        a (float): potential field parameter
        b (float): potential field parameter
        c (float): potential field parameter
        r0 (float): minimum distance to be part of the swarm
        delta (float): parameter for the neighbour weight

    Returns:
        float: combined forces for attraction/repulsion
    """
    
    return 1/r0 * get_neighbour_weight_der(r, r0, delta) * get_cohesion_intensity(r, d_ref, a, b, c) \
            + get_neighbour_weight(r, r0, delta) * get_cohesion_intensity_der(r, d_ref, a, b, c)

def get_migration_force(p_mig: np.ndarray[float], p_i: np.ndarray[float], v_ref: float, v_i: np.ndarray[float], gamma: float) -> np.ndarray[float]:
    """
    Calculate the migration force for the Olfati-Saber model to reach a target point

    Args:
        p_mig (np.ndarray[float]): the target point to reach [x, y, z]
        p_i (np.ndarray[float]): the current position of the drone [x, y, z]
        v_ref (float): the reference velocity
        v_i (np.ndarray[float]): the current velocity of the drone [vx, vy, vz]
        gamma (float): a tunable parameter

    Returns:
        np.ndarray[float]: the migration force
    """
    if p_mig is None:
        return 0
    d = np.linalg.norm(p_mig - p_i)
    u_i = 1/d * (p_mig - p_i)
    return gamma*d*(v_ref*u_i-v_i)

# Compute the olfati-saber swarm commands
def olfati_saber_input(drone_pose: np.ndarray, neighbour_poses: np.ndarray, obstacles: list["Obstacle"], p_mig: np.ndarray=None, params: dict=dict()) -> np.ndarray:
    """
    Compute the olfati-saber swarm commands

    Args:
        drone_pose (np.ndarray): current drone pose [x, y, z, vx, vy, vz, phi, theta, psi]
        neighbour_poses (np.ndarray): neighbour drone poses [x, y, z, vx, vy, vz, phi, theta, psi]
        obstacles (np.ndarray): obstacles definition (use Obstacle class)
        p_mig (np.ndarray, optional): target point to reach. Defaults to None.
        params (dict, optional): other defined parameters. Defaults to dict().

    Returns:
        np.ndarray: _description_
    """
    # AGENT PARAMETERS
    v_ref = params.get('v_ref',np.array([0.0,0.0,0.0]))
    d_ref = params.get('d_ref',1.0)
    r0_coh = params.get('r0_coh',20)
    delta = params.get('delta',0.1)
    a = params.get('a',0.3)
    b = params.get('b',0.5)
    c = params.get('c', (b - a)/(2*np.sqrt(a*b)))

    # OBSTACLES PARAMETERS
    a_obs = params.get('a_obs',0.3)
    b_obs = params.get('b_obs',0.5)
    c_obs = params.get('c_obs', (b_obs - a_obs)/(2*np.sqrt(a_obs*b_obs)))
    r0_obs = params.get('r0_coh_obs',0.8)
    d_ref_obs = params.get('d_ref_obs',0.75)
    lambda_obs = params.get('lambda_obs',1.0)
    c_pm_obs = params.get('c_pm_obs',6)
    c_vm_obs = params.get('c_vm_obs',0.5)

    # MIGRATION PARAMETERS
    v_ref_target = params.get('v_ref_target',0.5)
    c_vm = params.get('c_vm',0.3)
    gamma = params.get('gamma',1)
    
    #count += 1
    drone_pos = drone_pose[0]
    drone_vel = drone_pose[1]

    # Get the neighbour positions for neighbours above the ground
    neighbour_positions = [neighbour_pose[0] for neighbour_pose in neighbour_poses if neighbour_pose[0][2] > 0.1]
    num_neighbours = len(neighbour_poses)

    # Getting v_ref vector from vel_cmd
    v_ref_glob = rot_body2global(v_ref, drone_pose[3]) 

    # Normalize the reference velocity
    if np.linalg.norm(v_ref_glob) > 0:
        v_ref_u = v_ref_glob / np.linalg.norm(v_ref_glob)
    else:
        v_ref_u = v_ref_glob

    # Compute the velocity matching force
    acc_vel = np.zeros(3)
    if p_mig is None:
        acc_vel = c_vm * (v_ref_glob - drone_vel)
        
    # Initialize the cohesion command
    acc_coh = np.zeros(3)

    # Initialize the migration command
    acc_mig = np.zeros(3)

    # Compute the cohesion force for each neighbour
    if num_neighbours > 0:
        for neighbour_pos in neighbour_positions:
            
            # Get relative position and distance
            pos_rel = neighbour_pos - drone_pos
            dist = np.linalg.norm(pos_rel)

            # Compute the cohesion force
            acc_coh += get_cohesion_force(dist, d_ref, a, b, c, r0_coh, delta) * pos_rel / dist
            
        # TODO Verify if this is needed
        # Rotate the cohesion force to the body reference frame
        #acc_coh = rot_global2body(acc_coh, drone_pose[3])
    
    # Compute the migration force
    acc_mig += get_migration_force(p_mig, drone_pos, v_ref_target, drone_vel, gamma)
    #acc_mig = rot_global2body(acc_mig, drone_pose[3])

    # Initialize the obstacle avoidance commands
    acc_obs = np.zeros(3)

    # Compute the obstacle avoidance commands
    for obs in obstacles:

        # Check obstacle height w.r.t drone height
        if drone_pos[2] > obs.center[2] + obs.height/2:
            continue

        if type(obs) != Cylinder:
            # not implemented yet
            print("WARNING: Obstacle other than cylinders not implemented yet")
            continue
        
        # Extract the cylinder position
        cylinder_pos = obs.center
        cylinder_pos[2] = drone_pos[2]

        pos_rel = drone_pos - cylinder_pos
        dist = np.linalg.norm(pos_rel) - obs.radius
        
        if dist < r0_obs:

            # s in range (0,1]
            s = obs.radius / (dist + obs.radius)
            pos_obs = s*drone_pos + (1-s)*cylinder_pos

            # Derivative of s
            s_der = obs.radius * (drone_vel * (pos_obs - drone_pos) / dist) / (obs.radius + dist)**2
            
            vel_obs = s * drone_vel - obs.radius * (s_der/s) * (pos_obs-drone_pos)/dist
            pos_gamma = cylinder_pos + lambda_obs * v_ref_u
            d_ag = np.linalg.norm(pos_gamma - pos_obs)

            acc_obs += c_pm_obs * get_neighbour_weight(dist/r0_obs, r0_obs, delta) * (get_cohesion_force(dist, d_ref_obs, a_obs, b_obs, c_obs, r0_obs, delta)*(pos_obs - drone_pos)/dist +
                                    get_cohesion_force(d_ag, d_ref_obs, a_obs, b_obs, c_obs, r0_obs, delta)*(pos_gamma - drone_pos)/(np.linalg.norm(pos_gamma - drone_pos))) + c_vm_obs * (vel_obs - drone_vel)

    # # Rotate the obstacle avoidance force to the body reference frame
    # acc_obs = self.rot_global2body(acc_obs, drone_pose[2][2])

    # Remove the z component of the cohesion command
    #acc_coh[2] = 0        
    acc_command = acc_vel + acc_coh + acc_mig + acc_obs
    # ======================
    # !!! IMPORTANT !!!
    # NOT CONVERTING TO BODY FRAME
    # BECAUSE OF UNSTABILITY ISSUES
    # KEEP IT GLBOAL FRAME FOR SWARM MEMBERS
    #acc_command = rot_global2body(acc_command, drone_pose[3])

    return acc_command

