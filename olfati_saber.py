import numpy as np

'''
    This code is greatly inspired from the swarm_pilot.py file
    coded by Benjamin Jarvis for his doctoral thesis.
    All rights reserved
'''

# Calculate the cohesion intensity for the Olfati-Saber model
def get_cohesion_intensity(r, d_ref, a, b, c):
    
    diff = r - d_ref
    return ((a+b)/2 * (np.sqrt(1+(diff + c)**2) - np.sqrt(1+c**2)) + (a-b)*diff/2)

# Calculate the cohesion intensity derivative for the Olfati-Saber model
def get_cohesion_intensity_der(r, d_ref, a, b, c):
        
    diff = r - d_ref
    return (a+b)/2 * (diff + c) / np.sqrt(1+(diff + c)**2) + (a-b)/2

# Calcualte the neighbour weight for the Olfati-Saber model
def get_neighbour_weight(r, r0, delta):

    r_ratio = r / r0

    if r_ratio < delta:
        return 1
    elif r_ratio < 1:
        return 0.25 * (1 + np.cos(np.pi * (r_ratio - delta) / (1 - delta)))**2  #with k=2
    else:
        return 0

# Calcualte the derivative of the neighbour weight for the Olfati-Saber model
def get_neighbour_weight_der(r, r0, delta):
    
    r_ratio = r/r0

    if r_ratio < delta:
        return 0
    elif r_ratio < 1:
        arg = np.pi * (r_ratio - delta) / (1 - delta)
        return 1/2*(-np.pi/(1-delta))*(1+np.cos(arg))*np.sin(arg)
    else:
        return 0

# Calculate the attraction/repulsion force for the Olfati-Saber model
def get_cohesion_force(r, d_ref, a, b, c, r0, delta):
    
    return 1/r0 * get_neighbour_weight_der(r, r0, delta) * get_cohesion_intensity(r, d_ref, a, b, c) + get_neighbour_weight(r, r0, delta) * get_cohesion_intensity_der(r, d_ref, a, b, c)

def get_migration_force(p_mig, p_i, v_ref, v_i, gamma):
    if p_mig is None:
        return 0
    d = np.linalg.norm(p_mig - p_i)
    u_i = 1/d * (p_mig - p_i)
    return gamma*d*(v_ref*u_i-v_i)

# Compute the olfati-saber swarm commands
def olfati_saber_input(drone_pose, neighbour_poses, cylinder_poses, p_mig=None, **params):
    # Extract necessary params
    v_ref = params.get('v_ref',np.array([0.0,0.0,0.0]))
    d_ref = params.get('d_ref',1.0)
    r0_coh = params.get('r0_coh',20)
    delta = params.get('delta',0.1)
    a = params.get('a',0.3)
    b = params.get('b',0.5)
    c = params.get('c',(b - a)/(2*np.sqrt(a*b)))
    c_vm = params.get('c_vm',1)
    r0_obs = params.get('r0_obs',0.6)
    lambda_obs = params.get('lambda_obs',1)
    c_pm_obs = params.get('c_pm_obs',4.3)
    c_vm_obs = params.get('c_vm_obs',0)
    gamma = params.get('gamma',1)
    
    #count += 1
    drone_pos = drone_pose[0]
    drone_vel = drone_pose[1]

    # Get the neighbour positions for neighbours above the ground
    neighbour_positions = [neighbour_pose[0] for neighbour_pose in neighbour_poses if neighbour_pose[0][2] > 0.1]
    num_neighbours = len(neighbour_poses)

    # TODO Set reference velocity based either on a command input or migration point   

    # Normalize the reference velocity
    if np.linalg.norm(v_ref) > 0:
        v_ref_u = v_ref / np.linalg.norm(v_ref)
    else:
        v_ref_u = v_ref

    # Compute the velocity matching force
    acc_vel = np.zeros(3)
    if p_mig is None:
        acc_vel = c_vm * (v_ref - drone_vel)
        
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
        #acc_coh = rot_global2body(acc_coh, drone_pose[2][2])
    
    # Compute the migration force
    acc_mig += get_migration_force(p_mig, drone_pos, v_ref, drone_vel, gamma)

    # # Initialize the obstacle avoidance commands
    # acc_obs = np.zeros(3)

    # # Compute the obstacle avoidance commands
    # for i in range(self.num_obs):
        
    #     cylinder_poses[i][2] = drone_pos[2]

    #     pos_rel = drone_pos - cylinder_poses[i]
    #     dist = np.linalg.norm(pos_rel) - self.cylinder_radius
        
    #     if dist < r0_obs:

    #         # s in range (0,1]
    #         s = self.cylinder_radius / (dist + self.cylinder_radius)
    #         pos_obs = s*drone_pos + (1-s)*cylinder_poses[i]

    #         # Derivative of s
    #         s_der = self.cylinder_radius * (drone_vel * (pos_obs - drone_pos) / dist) / (self.cylinder_radius + dist)**2
            
    #         vel_obs = s * drone_vel - self.cylinder_radius * (s_der/s) * (pos_obs-drone_pos)/dist
    #         pos_gamma = cylinder_poses[i] + lambda_obs * v_ref_u
    #         d_ag = np.linalg.norm(pos_gamma - pos_obs)

    #         acc_obs += c_pm_obs * self.get_neighbour_weight(dist/r0_obs, r0_coh, delta) * (self.get_cohesion_force(dist, d_ref_obs, a, b, c, r0_coh, delta)*(pos_obs - drone_pos)/dist +
    #                                 self.get_cohesion_force(d_ag, d_ref_obs, a, b, c, r0_coh, delta)*(pos_gamma - drone_pos)/(np.linalg.norm(pos_gamma - drone_pos))) + c_vm_obs * (vel_obs - drone_vel)

    # # Rotate the obstacle avoidance force to the body reference frame
    # acc_obs = self.rot_global2body(acc_obs, drone_pose[2][2])

    # Remove the z component of the cohesion command
    #acc_coh[2] = 0        
    acc_command = acc_vel + acc_coh + acc_mig #+ acc_obs

    return acc_command