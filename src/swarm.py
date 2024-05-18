import numpy as np
from drone import *
import olfati_saber as olsab
from scipy.spatial import ConvexHull
from helper_functions import elapsed_timer

# Setting a common/uniform seed for testing
np.random.seed(1)

# Circle Trajectory
NB_POINTS = 30
TARGET_TOL = 0.2 # Tolerance before reaching the target
CIRCLE_RADIUS = 3.0
Z_HEIGHT = 5.0
t = np.linspace(-np.pi/2, 3*np.pi/2, NB_POINTS)
TRAJECTORY_CIRCLE = np.array([CIRCLE_RADIUS*np.cos(t), CIRCLE_RADIUS*np.sin(t), Z_HEIGHT*np.ones(NB_POINTS)]).T 
SCALE = 2
TRAJECTORY_INF_LOOP = np.array([SCALE*np.cos(t), SCALE*np.sin(2*t)/2, Z_HEIGHT*np.ones(NB_POINTS)]).T


MAX_ITER = 500 # Maximum number of iterations to find a valid position
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
        self.timing_neighborhood = 0.0
        self.timing_viewing_dir = 0.0
        self.timing_coverage = 0.0
        self.circle_done = False
        self.dist_weights = np.ones(self.count)
        # Initialize drones within a given box (random)
        if count == 1:
            self.members.append(Drone(init_pos=box[0:3]))
        else:
            box = np.array(box)
            pos = np.random.uniform(box[0:3] - box[3:6]/2, box[0:3] + box[3:6]/2, size=(count,3))
            for p in pos:
                self.members.append(Drone(init_pos=p, init_angles=[0,0,0], fov=360/count))
        #print("INITIALIZING SWARM: {0} drones within {1} box".format(count, box))
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
        if computation_method == 'Selected' or computation_method == 'All':
            if computation_method == 'Selected':
                with elapsed_timer() as elapsed:
                    self.members[self.selected_drone].compute_neihgborhood(self.members, metric, self.neighbors_params)
                    self.timing_neighborhood = elapsed()
                # Compute viewing direction
                if self.viewing_params.get('algorithm', 'None') != 'None':
                    self.members[self.selected_drone].compute_viewing_dir(self.members, self.viewing_params)
                    self.timing_viewing_dir = self.members[self.selected_drone].timing_viewing_dir
                only_selected = True
            elif computation_method == 'All':
                with elapsed_timer() as elapsed:
                    for m in self.members:
                        m.compute_neihgborhood(self.members, metric, self.neighbors_params)
                    self.timing_neighborhood = elapsed()
                
                self.timing_viewing_dir = 0
                for m in self.members:
                    if self.viewing_params.get('algorithm', 'None') != 'None':
                        m.compute_viewing_dir(self.members, self.viewing_params)
                        self.timing_viewing_dir += m.timing_viewing_dir
                only_selected = False
            # Compute coverage
            with elapsed_timer() as elapsed:
                self.compute_coverage(only_selected=only_selected)
                self.timing_coverage = elapsed()
        else:
            # clear all neighbors
            for m in self.members:
                m.neighbors = []

    def compute_coverage(self, only_selected=False):
        viewing_dir_2d = self.viewing_params.get('in_2d', False)
        # Compute distances of each drones to the convex hull of the swarm
        if self.is_2D:
            p_dim = 2
        else:
            p_dim = 3
        points = np.array([m.pos[:p_dim] for m in self.members])
        hull = ConvexHull(points)
        hull_center = np.mean(points[hull.vertices], axis=0)
        self.dist_weights = np.ones(self.count)
        for i in range(self.count):
            if i not in hull.vertices:
                # Find distance to closest edge
                dist = np.zeros(len(hull.simplices))
                for j in range(len(hull.simplices)):
                    idx = hull.simplices[j]
                    p = np.mean(points[idx], axis=1)
                    dist[j] = np.dot(p-points[i], hull.equations[j][:p_dim])
                idx_min = np.argmin(np.abs(dist))
                d_center = np.abs(np.dot(np.mean(points[hull.simplices[idx_min]], axis=1) - hull_center, hull.equations[idx_min][:p_dim]))
                self.dist_weights[i] = 1 - (dist[idx_min]/d_center)

        if viewing_dir_2d:
            # Compute the coverage of the swarm projected to a circle
            if only_selected:
                fov = self.members[self.selected_drone].fov
                coverage = fov
            else:
                nb_bins = int(2*np.pi/COVERAGE_RES) + 1
                coverage = np.zeros(nb_bins)
                for m in self.members:
                    fov = m.fov
                    psi = m.angles[2]
                    if psi < 0:
                        psi += 2*np.pi
                    if psi - fov/2 < 0:
                        l_bound = psi - fov/2 + 2*np.pi
                        u_bound = psi + fov/2
                        coverage[int(l_bound/COVERAGE_RES):] += 1
                        coverage[:int(u_bound/COVERAGE_RES)] += 1
                    elif psi + fov/2 > 2*np.pi:
                        l_bound = psi - fov/2
                        u_bound = psi + fov/2 - 2*np.pi
                        coverage[int(l_bound/COVERAGE_RES):] += 1
                        coverage[:int(u_bound/COVERAGE_RES)] += 1
                    else:
                        coverage[int((psi - fov/2)/COVERAGE_RES):int((psi + fov/2)/COVERAGE_RES)] += 1
                coverage = np.sum(np.clip(coverage, 0, 1))*COVERAGE_RES
            self.swarm_coverage = min(coverage/(2*np.pi), 1) # Clipping to 1.0 because of rounding errors caused by discretization
        else:
            # Compute the coverage of the swarm projected to a sphere
            # Only selected drone
            if only_selected:
                m = self.members[self.selected_drone]
                fov_psi = m.fov
                fov_phi = m.fov * m.ASPECT_RATIO
                coverage = fov_phi*fov_psi
            else:
                nb_bins_phi = int(np.pi/COVERAGE_RES) + 1
                nb_bins_psi = int(2*np.pi/COVERAGE_RES) + 1
                coverage = np.zeros((nb_bins_phi, nb_bins_psi))
                for m in self.members:
                    fov_psi = m.fov
                    fov_phi = m.fov * m.ASPECT_RATIO
                    psi = m.angles[2]
                    phi = m.angles[1] + np.pi/2
                    if psi < 0:
                        psi += 2*np.pi
                    if psi - fov_psi/2 < 0:
                        l_bound_psi = psi - fov_psi/2 + 2*np.pi
                        u_bound_psi = psi + fov_psi/2
                        if phi - fov_phi/2 < 0:
                            l_bound_phi = fov_phi/2 - phi
                            u_bound_phi = phi + fov_phi/2
                            l_bound_flipped = (psi - fov_psi/2 + np.pi) % (2*np.pi)
                            u_bound_flipped = (u_bound_psi + np.pi) % (2*np.pi)
                            coverage[:int(u_bound_phi/COVERAGE_RES), int(l_bound_psi/COVERAGE_RES):] += 1
                            coverage[:int(u_bound_phi/COVERAGE_RES), :int(u_bound_psi/COVERAGE_RES)] += 1
                            coverage[:int(l_bound_phi/COVERAGE_RES), int(l_bound_flipped/COVERAGE_RES):int(u_bound_flipped/COVERAGE_RES)] += 1
                        elif phi + fov_phi/2 > np.pi:
                            l_bound_phi = phi - fov_phi/2
                            u_bound_phi = 2*np.pi-(phi + fov_phi/2)
                            l_bound_flipped = (psi - fov_psi/2 + np.pi) % (2*np.pi)
                            u_bound_flipped = (u_bound_psi + np.pi) % (2*np.pi)
                            coverage[int(l_bound_phi/COVERAGE_RES):, int(l_bound_psi/COVERAGE_RES):] += 1
                            coverage[int(l_bound_phi/COVERAGE_RES):, :int(u_bound_psi/COVERAGE_RES)] += 1
                            coverage[int(u_bound_phi/COVERAGE_RES):, int(l_bound_flipped/COVERAGE_RES):int(u_bound_flipped/COVERAGE_RES)] += 1
                        else:
                            l_bound_phi = phi - fov_phi/2
                            u_bound_phi = phi + fov_phi/2
                            coverage[int(l_bound_phi/COVERAGE_RES):int(u_bound_phi/COVERAGE_RES), int(l_bound_psi/COVERAGE_RES):] += 1
                            coverage[int(l_bound_phi/COVERAGE_RES):int(u_bound_phi/COVERAGE_RES), :int(u_bound_psi/COVERAGE_RES)] += 1

                    elif psi + fov_psi/2 > 2*np.pi:
                        l_bound_psi = psi - fov_psi/2
                        u_bound_psi = psi + fov_psi/2 - 2*np.pi
                        if phi - fov_phi/2 < 0:
                            l_bound_phi = fov_phi/2 - phi
                            u_bound_phi = phi + fov_phi/2
                            l_bound_flipped = (l_bound_psi + np.pi) % (2*np.pi)
                            u_bound_flipped = (psi + np.pi + fov_psi/2) % (2*np.pi)
                            coverage[:int(u_bound_phi/COVERAGE_RES), int(l_bound_psi/COVERAGE_RES):] += 1
                            coverage[:int(u_bound_phi/COVERAGE_RES), :int(u_bound_psi/COVERAGE_RES)] += 1
                            coverage[:int(l_bound_phi/COVERAGE_RES), int(l_bound_flipped/COVERAGE_RES):int(u_bound_flipped/COVERAGE_RES)] += 1
                        elif phi + fov_phi/2 > np.pi:
                            l_bound_phi = phi - fov_phi/2
                            u_bound_phi = 2*np.pi-(phi + fov_phi/2)
                            l_bound_flipped = (l_bound_psi + np.pi) % (2*np.pi)
                            u_bound_flipped = (psi + np.pi + fov_psi/2) % (2*np.pi)
                            coverage[int(l_bound_phi/COVERAGE_RES):, int(l_bound_psi/COVERAGE_RES):] += 1
                            coverage[int(l_bound_phi/COVERAGE_RES):, :int(u_bound_psi/COVERAGE_RES)] += 1
                            coverage[int(u_bound_phi/COVERAGE_RES):, int(l_bound_flipped/COVERAGE_RES):int(u_bound_flipped/COVERAGE_RES)] += 1
                        else:
                            l_bound_phi = phi - fov_phi/2
                            u_bound_phi = phi + fov_phi/2
                            coverage[int(l_bound_phi/COVERAGE_RES):int(u_bound_phi/COVERAGE_RES), int(l_bound_psi/COVERAGE_RES):] += 1
                            coverage[int(l_bound_phi/COVERAGE_RES):int(u_bound_phi/COVERAGE_RES), :int(u_bound_psi/COVERAGE_RES)] += 1
                    else:
                        l_bound_psi = psi - fov_psi/2
                        u_bound_psi = psi + fov_psi/2
                        if phi - fov_phi/2 < 0:
                            l_bound_phi = fov_phi/2 - phi
                            u_bound_phi = phi + fov_phi/2
                            l_bound_flipped = (l_bound_psi + np.pi) % (2*np.pi)
                            u_bound_flipped = (u_bound_psi + np.pi) % (2*np.pi)
                            coverage[:int(u_bound_phi/COVERAGE_RES), int(l_bound_psi/COVERAGE_RES):int(u_bound_psi/COVERAGE_RES)+1] += 1
                            if u_bound_flipped < l_bound_flipped:
                                coverage[:int(l_bound_phi/COVERAGE_RES), int(l_bound_flipped/COVERAGE_RES):] += 1
                                coverage[:int(l_bound_phi/COVERAGE_RES), :int(u_bound_flipped/COVERAGE_RES)+1] += 1
                            else:
                                coverage[:int(l_bound_phi/COVERAGE_RES), int(l_bound_flipped/COVERAGE_RES):int(u_bound_flipped/COVERAGE_RES)+1] += 1
                        elif phi + fov_phi/2 > np.pi:
                            l_bound_phi = phi - fov_phi/2
                            u_bound_phi = 2*np.pi-(phi + fov_phi/2)
                            l_bound_flipped = (l_bound_psi + np.pi) % (2*np.pi)
                            u_bound_flipped = (u_bound_psi + np.pi) % (2*np.pi)
                            coverage[int(l_bound_phi/COVERAGE_RES):, int(l_bound_psi/COVERAGE_RES):int(u_bound_psi/COVERAGE_RES)+1] += 1
                            if u_bound_flipped < l_bound_flipped:
                                coverage[int(u_bound_phi/COVERAGE_RES):, int(l_bound_flipped/COVERAGE_RES):] += 1
                                coverage[int(u_bound_phi/COVERAGE_RES):, :int(u_bound_flipped/COVERAGE_RES)+1] += 1
                            else:
                                coverage[int(u_bound_phi/COVERAGE_RES):, int(l_bound_flipped/COVERAGE_RES):int(u_bound_flipped/COVERAGE_RES)+1] += 1
                        else:
                            l_bound_phi = phi - fov_phi/2
                            u_bound_phi = phi + fov_phi/2
                            coverage[int(l_bound_phi/COVERAGE_RES):int(u_bound_phi/COVERAGE_RES)+1, int(l_bound_psi/COVERAGE_RES):int(u_bound_psi/COVERAGE_RES)+1] += 1
                coverage = np.sum(np.clip(coverage, 0, 1))*COVERAGE_RES*COVERAGE_RES
            self.swarm_coverage = min(coverage/(2*np.pi*np.pi), 1)


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
                if self.trajectory_idx % NB_POINTS == 0:
                    self.circle_done = True
                self.migration_point = TRAJECTORY_INF_LOOP[self.trajectory_idx % NB_POINTS]

    def set_migration_mode(self, mode):
        self.migration_mode = mode
        if mode == 'trajectory':
            self.migration_point = TRAJECTORY_INF_LOOP[self.trajectory_idx]

    def update_drones_FOV(self, fov):
        for m in self.members:
            m.set_fov(fov)

    def set_count(self, count):
        diff = count - self.count
        if diff > 0:
            # Adding drones - stay within the bounding box
            # Shifted by the swarm center
            for i in range(int(diff)):
                pos_valid = False
                iter = 0
                while not pos_valid and iter < MAX_ITER:
                    r = np.random.rand()*self.algo_params.get('d_ref', 1.0)*3
                    theta = np.random.rand()*2*np.pi
                    phi = np.random.rand()*2*np.pi if not self.is_2D else np.pi/2
                    pos = self.swarm_center + np.array([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])
                    pos_valid = np.all(np.linalg.norm(np.array([self.members[i].pos for i in range(self.count)]) - pos, axis=1) > self.algo_params.get('d_ref', 1.0))
                    iter += 1
                if pos_valid:
                    self.members.append(Drone(init_pos=pos)) 
                else:
                    print("Could not find a valid position for the new drone!")
            self.count = len(self.members)
               

        elif diff < 0:
            # Removing drones
            choices = list(range(self.count))
            choices.remove(self.selected_drone)
            to_keep = [self.members[self.selected_drone]]
            for i in np.random.choice(choices, count-1, replace=False):
                to_keep.append(self.members[i])
            self.members = to_keep
            self.selected_drone = 0
            self.count = count
        
        self.update_drones_FOV(360/self.count)
        self.compute_neighborhood()

        