import numpy as np
from drone import *
import olfati_saber as olsab

# Setting a common/uniform seed for testing
np.random.seed(1)

class Swarm():
    def __init__(self, count=1, box=[0.0]*6, **kwargs):
        self.members = list()
        self.count = count
        self.migration_point = None
        self.noise = {'type': 'None', 'param_pos': 0.0, 'param_heading': 0.0}
        self.algo_params = {}
        self.ang_rates = np.zeros(3)
        self.selected_drone = 0
        self.update_counter = 0
        self.member_size = 0.025
        # Initialize drones within a given box (random)
        if count == 1:
            self.members.append(Drone(init_pos=box[0:3]))
        else:
            box = np.array(box)
            pos = np.random.uniform(box[0:3] - box[3:6]/2, box[0:3] + box[3:6]/2, size=(count,3))
            for p in pos:
                self.members.append(Drone(init_pos=p, init_angles=[0,0,0]))
        print("INITIALIZING SWARM: {0} drones within {1} box".format(count, box))

        # Intitialize optional parameters
        if 'migration_point' in kwargs:
            self.migration_point = kwargs['migration_point']
        if 'algo_params' in kwargs:
            self.algo_params = kwargs['algo_params']
        self.neighbors_params = kwargs.get('neighbors_metric', {'computation':'None', 'metric': 'Eucledian', 'sampling': 1})
        self.viewing_params = kwargs.get('viewing_metric', {'algorithm': 'None'})
        
    def update(self, dt, new_acc):
        for m in self.members:
            # Get the neighbors of member m
            neighborhood = self.get_neighbors(m)
            # Compute drone acceleration based on olfati-saber step
            neighbor_poses = [n.get_state() for n in neighborhood]
            new_acc = olsab.olfati_saber_input(m.get_state(), neighbor_poses, [], self.migration_point, self.algo_params)
            # Perform update step based on new acceleration
            m.update(dt, new_acc, self.ang_rates)
        # Increment the update counter (used for different purposes, e.g. sampling the computation of the neighborhood metric)
        self.update_counter += 1
        # Compute each drone neighborhood
        if self.update_counter % self.neighbors_params.get('sampling', 1) == 0:
            self.compute_neighborhood()

    def set_cmd_velocity(self, v_ref):
        self.algo_params['v_ref'] = v_ref

    def get_cmd_velocity(self):
        return self.algo_params.get('v_ref', np.zeros(3))
    
    def update_neighbors_metric(self, new_params):
        self.neighbors_params.update(new_params)

    def get_neighbor_metric(self, metric, default_val):
        if metric in self.neighbors_params:
            return self.neighbors_params[metric]
        else:
            return default_val

    def set_cmd_ang_rates(self, rates):
        if not np.all(rates == self.ang_rates):
            #print("Setting angular rates to: {0}".format(rates))
            self.ang_rates = rates

    def set_viewing_algorithm(self, algo):
        self.viewing_params['algorithm'] = algo

    def set_noise(self, type: str, param_dist:float, param_dir:float, apply_all=False):
        """ Setting the noise to sample when estimating the neigborhood of each drone.

        Args:
            type (str): Noise distribution type (None, Gaussian, Uniform)
            param_pos (float): Noise in position (computed based on distance and direction to neighbor) eg. sigma
            param_dir (float): Noise in sensing cone (direction)
        """
        if apply_all:
            for m in self.members:
                m.set_noise(type, param_dist, param_dir)
        else:
            self.members[self.selected_drone].set_noise(type, param_dist, param_dir)

    def get_noise(self):
        return self.members[self.selected_drone].noise

    def get_neighbors(self, d:Drone, metric='Olfati-Saber') -> list[Drone]:
        '''
        Function that returns neighbors based on the desired metric.
        '''
        match metric:
            case "Olfati-Saber":
                # Use all neighbors, internal algorithm automatically weight them
                m = self.members.copy()
                m.remove(d)
                return m
            case _:
                return list()
    
    def compute_neighborhood(self):
        computation_method = self.neighbors_params.get('computation', 'None')
        metric = self.neighbors_params.get('metric', 'Eucledian')
        if computation_method == 'Selected':
            self.members[self.selected_drone].compute_neihgborhood(self.members, metric, self.neighbors_params)
            # Compute viewing direction
            if self.viewing_params.get('algorithm', 'None') != 'None':
                self.members[self.selected_drone].compute_viewing_dir(self.members, self.viewing_params.get('algorithm', 'None'))
        elif computation_method == 'All':
            for m in self.members:
                m.compute_neihgborhood(self.members, metric, self.neighbors_params)
                if self.viewing_params.get('algorithm', 'None') != 'None':
                    m.compute_viewing_dir(self.members, self.viewing_params.get('algorithm', 'None'))
        else:
            # clear all neighbors
            for m in self.members:
                m.neighbors = []

    def initialize_random_vel(self, bounds):
        for m in self.members:
            for i in range(3):
                m.vel[i] = bounds[2*i] + (bounds[2*i+1] - bounds[2*i])*np.random.rand()

    def print_swarm(self):
        for i in range(self.count):
            print("====| DRONE {0} |=====".format(i+1))
            self.members[i].print_state()

    def get_states(self):
        if self.count == 0:
            return np.empty((1,12))
        states = np.zeros((self.count,12))
        for i in range(self.count):
            states[i,:] = self.members[i].get_state().reshape(1,12)
            states[i,9:] = self.members[i].get_heading()
        return states
