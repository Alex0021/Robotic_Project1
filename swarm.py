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
        # Initialize drones within a given box (random)
        if count == 1:
            self.members.append(Drone(init_pos=box[0:3]))
        else:
            for _ in range(count):
                x = box[0] + box[3]*(np.random.rand()-0.5)
                y = box[1] + box[4]*(np.random.rand()-0.5)
                z = box[2] + box[5]*(np.random.rand()-0.5)
                self.members.append(Drone(init_pos=[x,y,z], init_angles=[0,0,0]))
        print("INITIALIZING SWARM: {0} drones within {1} box".format(count, box))

        # Intitialize optional parameters
        if 'migration_point' in kwargs:
            self.migration_point = kwargs['migration_point']
        if 'noise' in kwargs:
            self.noise = kwargs['noise']
        if 'algo_params' in kwargs:
            self.algo_params = kwargs['algo_params']
        self.neighbors_metric = kwargs.get('neighbors_metric', {'computation':'None', 'metric': 'Eucledian', 'sampling': 1, 'metric_specific': {}})
        
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
        computation_method = self.neighbors_metric.get('computation', 'None')
        if computation_method != 'None':
            if (self.update_counter % self.neighbors_metric.get('sampling', 1)) == 0:
                # Only execute the computation every N steps
                metric = self.neighbors_metric.get('metric', 'Eucledian')
                metric_data = self.neighbors_metric.get('metric_specific', {})
                self.compute_neighborhood(metric, metric_data, computation_method)

    def set_cmd_velocity(self, v_ref):
        self.algo_params['v_ref'] = v_ref

    def get_cmd_velocity(self):
        return self.algo_params.get('v_ref', np.zeros(3))
    
    def update_neighbors_metric(self, new_params, apply_to_all):
        self.neighbors_metric.update(new_params)

    def set_cmd_ang_rates(self, rates):
        if not np.all(rates == self.ang_rates):
            #print("Setting angular rates to: {0}".format(rates))
            self.ang_rates = rates

    def set_noise(self, type: str, param_pos:float, param_heading:float):
        """ Setting the noise to sample when estimating the neigborhood of each drone.

        Args:
            type (str): Noise distribution type (None, Gaussian, Uniform)
            param_pos (float): Noise in position (computed based on distance and direction to neighbor) eg. sigma
            param_heading (float): Noise in heading (added to true heading of neighbor)
        """
        self.noise = {'type': type, 'param_pos': param_pos, 'param_heading': param_heading}

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
    
    def compute_neighborhood(self, metric, metric_data=dict(), computation_method='Selected'):
        if computation_method == 'Selected':
            self.members[self.selected_drone].compute_neihgborhood(self.members, metric, self.noise, metric_data)
        elif computation_method == 'All':
            for m in self.members:
                m.compute_neihgborhood(self.members, metric, self.noise, metric_data)
                if metric == "Voronoi":
                    break
                    # only one drone needs to compute the voronoi neighbors
        else:
            pass

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
