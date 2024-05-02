import numpy as np
from drone import *
import olfati_saber as olsab
from scipy.spatial import ConvexHull

# Setting a common/uniform seed for testing
np.random.seed(1)

# Circle Trajectory
NB_POINTS = 20
TARGET_TOL = 0.15 # Tolerance before reaching the target
CIRCLE_RADIUS = 3.0
TRAJECTORY_CIRCLE = np.array([CIRCLE_RADIUS*np.cos(np.linspace(0, 2*np.pi, NB_POINTS)), CIRCLE_RADIUS*np.sin(np.linspace(0, 2*np.pi, NB_POINTS)), 5.0*np.ones(NB_POINTS)]).T 

MAX_ITER = 100 # Maximum number of iterations to find a valid position
COVERAGE_RES = 0.01  # To discretize into cells for the coverage computation

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
        self.migration_mode = 'single' # ['single', 'trajectory']
        self.trajectory_idx = 0
        self.is_2D = abs(box[5]) < 0.01
        self.viewing_dim = 2 if self.is_2D else 3
        self.swarm_coverage = 0.0
        # Initialize drones within a given box (random)
        if count == 1:
            self.members.append(Drone(init_pos=box[0:3]))
        else:
            box = np.array(box)
            pos = np.random.uniform(box[0:3] - box[3:6]/2, box[0:3] + box[3:6]/2, size=(count,3))
            for p in pos:
                self.members.append(Drone(init_pos=p, init_angles=[0,0,0]))
        print("INITIALIZING SWARM: {0} drones within {1} box".format(count, box))
        self.swarm_center = self.get_swarm_center()

        # Intitialize optional parameters
        if 'migration_point' in kwargs:
            self.migration_point = kwargs['migration_point']
        if 'algo_params' in kwargs:
            self.algo_params = kwargs['algo_params']
        self.neighbors_params = kwargs.get('neighbors_metric', {'computation':'None', 'metric': 'Eucledian', 'sampling': 1})
        self.viewing_params = kwargs.get('viewing_metric', {'algorithm': 'None'})
        self.viewing_params.update({'in_2d': self.is_2D})

        # First trajectory point
        if self.migration_mode == 'trajectory':
            self.migration_point = TRAJECTORY_CIRCLE[self.trajectory_idx]
        
    def update(self, dt, new_acc):
        new_acc = np.zeros((self.count, 3))
        for i in range(self.count):
            m = self.members[i]
            # Get the neighbors of member m
            neighborhood = self.get_neighbors(m)
            # Compute drone acceleration based on olfati-saber step
            neighbor_poses = np.array([n.get_state() for n in neighborhood])
            new_acc[i,:] = olsab.olfati_saber_input(m.get_state(), neighbor_poses, [], self.migration_point, self.algo_params)
        # Perform update step based on new acceleration
        for i in range(self.count):
            self.members[i].update(dt, new_acc[i], self.ang_rates)
        # Increment the update counter (used for different purposes, e.g. sampling the computation of the neighborhood metric)
        self.update_counter += 1
        # Compute each drone neighborhood
        if self.update_counter % self.neighbors_params.get('sampling', 1) == 0:
            self.compute_neighborhood()
        # Check trajectory
        if self.migration_mode == 'trajectory':
            self.compute_next_target()

        self.swarm_center = self.get_swarm_center()

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

    def set_viewing_algorithm(self, algo, params=dict()):
        self.viewing_params['algorithm'] = algo
        self.viewing_params.update(params)

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
                self.members[self.selected_drone].compute_viewing_dir(self.members, self.viewing_params)
            # Compute coverage
            self.compute_coverage(only_selected=False)
        elif computation_method == 'All':
            for m in self.members:
                m.compute_neihgborhood(self.members, metric, self.neighbors_params)
                if self.viewing_params.get('algorithm', 'None') != 'None':
                    m.compute_viewing_dir(self.members, self.viewing_params)
            # Compute coverage
            self.compute_coverage(only_selected=False)
        else:
            # clear all neighbors
            for m in self.members:
                m.neighbors = []

    def compute_coverage(self, only_selected=False):
        viewing_dir_2d = self.viewing_params.get('in_2d', False)
        # Compute distances of each drones to the convex hull of the swarm
        weights = np.ones(self.count)
        if self.is_2D:
            points = np.array([m.pos[:2] for m in self.members])
            hull = ConvexHull(points)
            for i in range(self.count):
                if i not in hull.vertices:
                    # Find distance to closest edge
                    dist = np.zeros(len(hull.simplices))
                    for j in range(len(hull.simplices)):
                        idx = hull.simplices[j][0]
                        p = points[idx]
                        dist[j] = np.dot(p-points[i], hull.equations[j][:2])
                    idx_min = np.argmin(np.abs(dist))
                    d_center = np.abs(np.dot(points[hull.simplices[idx_min][0]] - self.swarm_center[:2], hull.equations[idx_min][:2]))
                    weights[i] = 1 - (dist[idx_min]/d_center)

        if viewing_dir_2d:
            # Compute the coverage of the swarm projected to a circle
            if only_selected:
                fov = self.members[self.selected_drone].fov
                coverage = fov
            else:
                nb_bins = int(2*np.pi/COVERAGE_RES) + 1
                coverage = np.zeros(nb_bins)
                for m, w in zip(self.members, weights):
                    fov = m.fov
                    psi = m.angles[2]
                    if psi < 0:
                        psi += 2*np.pi
                    if psi - fov/2 < 0:
                        l_bound = psi - fov/2 + 2*np.pi
                        u_bound = psi + fov/2
                        coverage[int(l_bound/COVERAGE_RES):] += w
                        coverage[:int(u_bound/COVERAGE_RES)] += w
                    elif psi + fov/2 > 2*np.pi:
                        l_bound = psi - fov/2
                        u_bound = psi + fov/2 - 2*np.pi
                        coverage[int(l_bound/COVERAGE_RES):] += w
                        coverage[:int(u_bound/COVERAGE_RES)] += w
                    else:
                        coverage[int((psi - fov/2)/COVERAGE_RES):int((psi + fov/2)/COVERAGE_RES)] += w
                coverage = len(np.where(coverage == 1)[0])*COVERAGE_RES
                # coverage = []
                # # Discretizing each fov
                # for m in self.members:
                #     fov = m.fov
                #     psi = m.angles[2]
                #     if psi < 0:
                #         psi += 2*np.pi
                #     if psi - fov/2 < 0:
                #         l_bound = psi - fov/2 + 2*np.pi
                #         u_bound = psi + fov/2
                #         coverage.append(np.arange(l_bound, 2*np.pi, COVERAGE_RES))
                #         coverage.append(np.arange(0, u_bound, COVERAGE_RES))
                #     elif psi + fov/2 > 2*np.pi:
                #         l_bound = psi - fov/2
                #         u_bound = psi + fov/2 - 2*np.pi
                #         coverage.append(np.arange(l_bound, 2*np.pi, COVERAGE_RES))
                #         coverage.append(np.arange(0, u_bound, COVERAGE_RES))
                #     else:
                #         coverage.append(np.arange(psi - fov/2, psi + fov/2, COVERAGE_RES))
                # # Compute ratio
                # coverage = len(np.unique(np.concatenate(coverage).round(2)))*COVERAGE_RES
        else:
            coverage = 0
        self.swarm_coverage = np.clip(coverage/(2*np.pi), 0, 1) # Because of rounding errors caused by discretization


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
    
    def get_swarm_center(self):
        return np.mean([m.pos for m in self.members], axis=0)
    
    def use_pd_smoothing(self, val):
        for m in self.members:
            m.use_pd_controller = val

    def compute_next_target(self):
        # Check if migration point is reached
        if self.migration_point is not None:
            if np.linalg.norm(self.swarm_center - self.migration_point) < TARGET_TOL:
                self.trajectory_idx += 1
                self.migration_point = TRAJECTORY_CIRCLE[self.trajectory_idx % NB_POINTS]

    def set_migration_mode(self, mode):
        self.migration_mode = mode
        if mode == 'trajectory':
            self.migration_point = TRAJECTORY_CIRCLE[self.trajectory_idx]

    def set_count(self, count):
        diff = count - self.count
        if diff > 0:
            # Adding drones - stay within the bounding box
            # Shifted by the swarm center
            for i in range(int(diff)):
                pos_valid = False
                iter = 0
                while not pos_valid and iter < MAX_ITER:
                    r = np.random.rand()*self.algo_params.get('d_ref', 1.0)*2
                    theta = np.random.rand()*2*np.pi
                    phi = np.random.rand()*2*np.pi if not self.is_2D else np.pi/2
                    pos = self.swarm_center + np.array([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])
                    pos_valid = np.all(np.linalg.norm(np.array([self.members[i].pos for i in range(self.count)]) - pos, axis=1) > 0.5*self.algo_params.get('d_ref', 1.0))
                    iter += 1
                if pos_valid:
                    self.members.append(Drone(init_pos=pos)) 
                    self.count = count
                else:
                    print("Could not find a valid position for the new drone!")   

        elif diff < 0:
            # Removing drones
            nb = int(abs(diff))
            choices = list(range(self.count))
            choices.pop(self.selected_drone)
            for i in np.random.choice(choices, nb, replace=False):
                self.members.pop(i)
            self.count = count
            self.compute_neighborhood()