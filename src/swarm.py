import numpy as np
from drone import *
import olfati_saber as olsab
from scipy.spatial import ConvexHull
from helper_functions import elapsed_timer
import typing

#===============================================================================
# Trajectory parameters
#===============================================================================
NB_POINTS = 50
TARGET_TOL = 0.25 # Tolerance before reaching the target
CIRCLE_RADIUS = 3.0
Z_HEIGHT = 5.0  # Cosntant height for the trajectory
SCALE = 2  # Scaling of the inf loop
t = np.linspace(-np.pi/2, 3*np.pi/2, NB_POINTS)
TRAJECTORY_CIRCLE = np.array([CIRCLE_RADIUS*np.cos(t), CIRCLE_RADIUS*np.sin(t), Z_HEIGHT*np.ones(NB_POINTS)]).T 
TRAJECTORY_INF_LOOP = np.array([SCALE*np.cos(t), SCALE*np.sin(2*t)/2, Z_HEIGHT*np.ones(NB_POINTS)]).T

#===============================================================================
# Swarm parameters
#===============================================================================
# Setting a common/uniform seed for testing
#np.random.seed(1)
STABILITY_SPEED_TOL = 0.015  # When the swarm is considered stabilized
MIN_STABILITY_DELAY = 1.0 # In seconds
MAX_ITER = 10000 # Maximum number of iterations to find a valid position
COVERAGE_RES = 0.01  # To discretize into cells for the coverage computation

class Swarm():
    """
    Class representing a swarm of drones.

    Responsible for:
      - managing the drones
      - updating their states
      - computing the neighborhood
      - calculations of swarm metrics (e.g. coverage)
    """
    def __init__(self, count: int=1, area: list[float]=[0.0,0.0,0.0,1.0], **kwargs):
        # Swarm parameters
        self.count = count
        self.members = list()
        self.migration_point = None
        self.noise = {'type': 'None', 'param_pos': 0.0, 'param_heading': 0.0}
        self.ang_rates = np.zeros(3)
        self.selected_drone = 0
        self.member_size = 0.025
        self.is_2D = kwargs.get('is_2d', True)
        self.spawn_area = area
        self.migration_mode = 'single' # ['single', 'trajectory']
        if 'migration_point' in kwargs:
            self.migration_point = kwargs['migration_point']
        self.algo_params = {}
        if 'algo_params' in kwargs:
            self.algo_params = kwargs['algo_params']
        self.neighbors_params = kwargs.get('neighbors_metric', {'computation':'None', 'metric': 'Eucledian', 'sampling': 1})
        self.viewing_params = kwargs.get('viewing_metric', {'algorithm': 'None'})
        self.viewing_params.update({'in_2d': self.is_2D})

        # Sim variables
        self.update_counter = 0
        self.trajectory_idx = 0
        self.circle_done = False
        self._stabilized = False

        # Performance metrics variables
        self.swarm_coverage = 0.0
        self.swarm_overlap = 0.0
        self.timing_neighborhood = 0.0
        self.timing_viewing_dir = 0.0
        self.timing_coverage = 0.0
        self.sim_time = 0.0
        self.dist_weights = np.ones(self.count)

        # Initialize trajectory points
        traj_type = kwargs.get('trajectory', 'circle')
        match traj_type:
            case 'circle':
                self.trajectory_points = TRAJECTORY_CIRCLE
            case 'inf_loop':
                self.trajectory_points = TRAJECTORY_INF_LOOP
            case _:
                raise ValueError("Invalid trajectory type: {0}".format(traj_type))

        # Setting first trajectory point
        if self.migration_mode == 'trajectory':
            self.migration_point = self.trajectory_points[self.trajectory_idx]

    #=======================================#
    #            Initialization             #
    #=======================================#
        
    def initialize_members(self):
        """
        Initialize the drones in the swarm with the given parameters.

        Raises:
            ValueError: If it fails to find a valid initial position for all drones.
        """
        rho = self.spawn_area[3]
        if self.is_2D:
            # Calculate the radius of the circle
            r_max = np.sqrt(self.count/(np.pi*rho))
        else:
            # calculate the radius of the sphere
            r_max = np.cbrt(3*self.count/(4*np.pi*rho))
        pos_hist = []
        for i in range(self.count):
            pos_valid = False
            iter = 0
            while not pos_valid and iter < MAX_ITER:
                r = r_max*np.random.rand()
                theta = np.random.rand()*2*np.pi
                phi = np.pi/2 if self.is_2D else np.random.rand()*np.pi
                pos = np.array([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)]) + np.array(self.spawn_area[0:3])
                # First position is always valid
                if i > 0:
                    pos_valid = np.all(np.linalg.norm(np.array([pos_hist[j] for j in range(i)]) - pos, axis=1) > 2*self.neighbors_params.get('r_agent', 0.069))
                else:
                    pos_valid = True
                iter += 1
            if pos_valid:
                pos_hist.append(pos)
                self.members.append(Drone(init_pos=pos, fov=360/self.count, swarm_2d=self.is_2D))
            else:
                raise ValueError("Could not find a valid itnial position for all drones in the swarm!")
        self.swarm_center = self.get_swarm_center()

    def initialize_random_vel(self, bounds):
        for m in self.members:
            for i in range(3):
                m.vel[i] = bounds[2*i] + (bounds[2*i+1] - bounds[2*i])*np.random.rand()


    #=======================================#
    #            Swarm dynamics             #
    #=======================================#

    def update(self, dt: float):
        """
        Performs update step for each drone in the swarm.

        Automatically called by the simulator.

        Args:
            dt (float): Time step for the update.
        """
        new_acc = np.zeros((self.count, 3))
        for i in range(self.count):
            m = self.members[i]
            # Get the neighbors of member m
            neighborhood = self.get_neighbors(m)
            # Compute drone acceleration based on selected algorithm
            match self.algo_params.get('algorithm', 'None').upper():
                case "NONE":
                    new_acc[i,:] = np.zeros(3)
                case "OLFATI-SABER":
                    neighbor_poses = np.array([n.get_state() for n in neighborhood])
                    new_acc[i,:] = olsab.olfati_saber_input(m.get_state(), neighbor_poses, [], self.migration_point, self.algo_params)
                case _:
                    raise ValueError("Invalid algorithm: {0}".format(self.algo_params.get('algorithm', 'None')))
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
        self.sim_time += dt

    def compute_neighborhood(self):
        """
        Compute the neighborhood of each drone in the swarm.

        Steps:
            - 1. Compute the neighborhood of each drone or only the selected one.
            - 2. Compute the viewing direction of each drone or only the selected one.
            - 3. Compute the coverage of the swarm.
        """
        computation_method = self.neighbors_params.get('computation', 'None').upper()
        metric = self.neighbors_params.get('metric', 'Eucledian')
        if computation_method == 'SELECTED' or computation_method == 'ALL':
            if computation_method == 'SELECTED':
                with elapsed_timer() as elapsed:
                    self.members[self.selected_drone].compute_neihgborhood(self.members, metric, self.neighbors_params)
                    self.timing_neighborhood = elapsed()
                # Compute viewing direction
                if self.viewing_params.get('algorithm', 'None').upper() != 'NONE':
                    self.members[self.selected_drone].compute_viewing_dir(self.viewing_params)
                    self.timing_viewing_dir = self.members[self.selected_drone].timing_viewing_dir
                only_selected = True
            elif computation_method == 'ALL':
                with elapsed_timer() as elapsed:
                    for m in self.members:
                        m.compute_neihgborhood(self.members, metric, self.neighbors_params)
                    self.timing_neighborhood = elapsed()
                
                self.timing_viewing_dir = 0
                for m in self.members:
                    if self.viewing_params.get('algorithm', 'None').upper() != 'NONE':
                        self.compute_ground_truth_viewing_dir()
                        m.compute_viewing_dir(self.viewing_params)
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

    def compute_next_target(self):
        """
        Compute the next target for the migration of the swarm.

        Modulus operation is used to loop over the trajectory points.
        """
        # Check if migration point is reached
        if not self._stabilized:
                self._stabilized = self.is_swarm_stabilized()
        elif self.migration_point is not None:
            if np.linalg.norm(self.swarm_center - self.migration_point) < TARGET_TOL:
                self.trajectory_idx += 1
                if self.trajectory_idx % NB_POINTS == 0:
                    self.circle_done = True
                self.migration_point = self.trajectory_points[self.trajectory_idx % NB_POINTS]

    #=======================================#
    #            Swarm metrics              #
    #=======================================#

    def compute_ground_truth_viewing_dir(self):
        """
            Compute the "best" viewing direction based on convex hull of the swarm

            Metric not 100% accurate, but gives a good baseline for the performance of the viewing algorithms
        """
        # Compute distances of each drones to the convex hull of the swarm
        if self.is_2D:
            p_dim = 2
        else:
            p_dim = 3
        points = np.array([m.pos[:p_dim] for m in self.members])
        hull = ConvexHull(points)
        hull_center = np.mean(points[hull.vertices], axis=0)
        self.dist_weights = np.ones(self.count)
        # Calculate first viewing dir on the convex hull
        for i in hull.vertices:
            # Drone is on the convex hull
            # Use average normals of connected edges/faces
            idx = np.where(hull.simplices == i)[0]
            normals = np.zeros((len(idx), p_dim))
            for j in range(len(idx)):
                normals[j] = hull.equations[idx[j]][:p_dim]
            vd = np.mean(normals, axis=0)
            self.members[i].ground_truth_viewing_dir[:p_dim] = vd / np.linalg.norm(vd)
        for i in np.setdiff1d(range(self.count), hull.vertices, assume_unique=True):
            # Find distance to closest edge
            dist = np.zeros(len(hull.simplices))
            for j in range(len(hull.simplices)):
                idx = hull.simplices[j]
                p = np.mean(points[idx], axis=0)
                eq_norm = hull.equations[j][:p_dim] / np.linalg.norm(hull.equations[j][:p_dim])
                dist[j] = np.dot(p-points[i], eq_norm)
            idx_min = np.argmin(np.abs(dist))
            view_dir = np.mean([self.members[k].ground_truth_viewing_dir for k in hull.simplices[idx_min]], axis=0)
            self.members[i].ground_truth_viewing_dir = view_dir / np.linalg.norm(view_dir)
            eq_norm = hull.equations[idx_min][:p_dim] / np.linalg.norm(hull.equations[idx_min][:p_dim])
            d_center = np.abs(np.dot(np.mean(points[hull.simplices[idx_min]]) - hull_center, eq_norm))
            self.dist_weights[i] = np.clip(1 - (dist[idx_min]/d_center), 0, 1)
        

    def compute_coverage(self, only_selected: bool=False):
        """
        Compute the coverage of the swarm. It is done by discretizing the space into cells and counting the number of cells covered by the drones.
        
        The overlap is a derivation of the coverage metric, it is the number of cells covered by more than one drone.

        In 2D:
            - Use the idea of an infinite radius circle and project the drones' FOV to it.

        In 3D:
            - Same principle as in 2D but project to a sphere.
            - Double discretization needed for both phi and psi angles.

        Args:
            only_selected (bool, optional): Only use the selected drone in the computation. Defaults to False.
        """
        viewing_dir_2d = self.viewing_params.get('in_2d', False)

        if viewing_dir_2d:
            # Compute the coverage of the swarm projected to a circle
            if only_selected:
                fov = self.members[self.selected_drone].fov
                coverage = fov
                self.swarm_overlap = 0
            else:
                nb_bins = int(2*np.pi/COVERAGE_RES) + 1
                coverage = np.zeros(nb_bins)
                for m in self.members:
                    fov = m.fov
                    # Readjusting the angle to be in [0, 2*pi]
                    psi = m.angles[2] + np.pi
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
                self.swarm_overlap = np.sum(np.clip(coverage-1, 0, np.inf))*COVERAGE_RES
                coverage = np.sum(np.clip(coverage, 0, 1))*COVERAGE_RES
            self.swarm_coverage = min(coverage/(2*np.pi), 1) # Clipping to 1.0 because of rounding errors caused by discretization
        else:
            #===============================================================================
            # More complicated in 3D
            # More cases to consider for the FOV with both angles
            #===============================================================================
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
                    # readjusting the angles to be in [0, 2*pi]
                    psi = m.angles[2] + np.pi
                    phi = m.angles[1] + np.pi/2

                    # Treat the case where the FOV overflows under 0 (discontinuity)
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

                    # Treat the case where the FOV overflows 2*pi (discontinuity)
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
                    
                    # Normal case (for psi angle)
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

    #=======================================#
    #        Getters and Setters            #
    #=======================================#

    def set_pd_controller(self, active: bool, kp: float, kd: float, rate_limit: float, **kwargs):
        """
        Set the PD controller parameters for all drones in the swarm.

        Args:
            active (bool): Flag to activate the PD controller.
            kp (float): Proportional gain.
            kd (float): Derivative gain.
            rate_limit (float): Rate limit for the controller.
        """
        for m in self.members:
            m.set_pd_controller(active, kp, kd, rate_limit)

    def set_cmd_velocity(self, v_ref: np.ndarray):
        """
        Set the reference velocity for the drones in the swarm.

        Args:
            v_ref (np.ndarray): Reference velocity in the form [vx, vy, vz].
        """
        self.algo_params['v_ref'] = v_ref

    def get_cmd_velocity(self) -> np.ndarray[float]:
        """
        Get the reference velocity for the drones in the swarm.

        Returns:
            np.ndarray[float]: Reference velocity in the form [vx, vy, vz].
        """
        return self.algo_params.get('v_ref', np.zeros(3))
    
    def set_cmd_ang_rates(self, rates: np.ndarray[float]):
        """
        Set the reference angular rates for the drones in the swarm.

        Args:
            rates (np.ndarray[float]):  Reference angular rates in the form [p, q, r].
        """
        if not np.all(rates == self.ang_rates):
            self.ang_rates = rates

    def get_cmd_ang_rates(self) -> np.ndarray[float]:
        """
        Get the reference angular rates for the drones in the swarm.

        Returns:
            np.ndarray[float]: Reference angular rates in the form [p, q, r].
        """
        return self.ang_rates
    
    def update_neighbors_metric(self, new_params: dict):
        """
        Update the parameters of the neighborhood metric.

        Args:
            new_params (dict): New parameters to be updated as a dictionnary.
        """
        self.neighbors_params.update(new_params)

    def get_neighbor_metric(self, metric: str, default_val: typing.Any) -> typing.Any:
        """_summary_

        Args:
            metric (str): dictionnary key to be retrieved.
            default_val (Any): Default value to be returned if the key is not found.

        Returns:
            Any: The value of the key in the dictionnary or the default value.
        """
        if metric in self.neighbors_params:
            return self.neighbors_params[metric]
        else:
            return default_val

    def set_swarming_algorithm(self, algo: str):
        """
        Set the swarming algorithm to be used.

        Args:
            algo (str): Name of the algorithm to be used.
        """
        self.algo_params['algorithm'] = algo

    def set_viewing_algorithm(self, algo: str, params: dict=dict()):
        """
        Set the viewing algorithm to be used.

        Args:
            algo (str): Name of the algorithm to be used.
            params (dict, optional): Parameters of the viewing algorithm. Defaults to empty dict.
        """
        self.viewing_params['algorithm'] = algo
        self.viewing_params.update(params)

    def set_noise(self, type: str, param_dist:float, param_dir:float, param_heading: float, apply_all=False):
        """ Setting the noise to sample when estimating the neigborhood of each drone.

        Args:
            type (str): Noise distribution type (None, Gaussian, Uniform)
            param_pos (float): Noise in position (computed based on distance and direction to neighbor) eg. sigma
            param_dir (float): Noise in sensing cone (direction)
            param_heading (float): Heading noise (in radians)
        """
        if apply_all:
            for m in self.members:
                m.set_noise(type, param_dist, param_dir, param_heading)
        else:
            self.members[self.selected_drone].set_noise(type, param_dist, param_dir, param_heading)

    def get_noise(self) -> dict:
        """
        Get the noise parameters of the selected drone.

        Returns:
            dict: Dictionary containing the noise parameters.
        """
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

    def get_states(self) -> np.ndarray:
        """
        Get the states of all drones in the swarm.

        Returns:
            np.ndarray: Array containing the states of all drones. Concatenated in the form [pos, vel, acc, heading]. Each row represents a drone.
        """
        if self.count == 0:
            return np.empty((1,12))
        states = np.zeros((self.count,12))
        for i in range(self.count):
            states[i,:] = self.members[i].get_state().reshape(1,12)
            states[i,9:] = self.members[i].get_heading()
        return states
    
    def get_swarm_center(self) -> np.ndarray[float]:
        """
        Get the center of the swarm. Simple average of the positions of all drones.

        Returns:
            np.ndarray[float]: Position of the swarm center [px, py, pz].
        """
        return np.mean([m.pos for m in self.members], axis=0)

    def is_swarm_stabilized(self) -> bool:
        """
        Check if the swarm is stabilized. 

        It is considered stabilized if all drones have a speed below a certain threshold 
        and the swarm has been stable for a certain amount of time.

        Returns:
            bool: True if the swarm is stabilized, False otherwise.
        """
        speeds = np.array([np.linalg.norm(m.vel) for m in self.members])
        return np.all(speeds < STABILITY_SPEED_TOL) and self.sim_time > MIN_STABILITY_DELAY

    def set_migration_mode(self, mode: str):
        """
        Set the migration mode of the swarm.

        Single: Migrate to a single point and stay there.

        Trajectory: Migrate to a set of points in a trajectory. 
                    Waypoints are automatically updated.

        Args:
            mode (str): Mode of migration ('single', 'trajectory').
        """
        self.migration_mode = mode
        if mode == 'trajectory':
            self.migration_point = self.trajectory_points[self.trajectory_idx]

    def update_drones_FOV(self, fov: float):
        """
        Update the field of view of all drones in the swarm.

        Args:
            fov (float): New field of view in degrees.
        """
        for m in self.members:
            m.set_fov(fov)

    def set_count(self, count: int):
        """
        Change the number of drones in the swarm.

        New drones added are placed in the vicinity of the swarm center.

        Drones removed are randomly selected.

        Args:
            count (int): New number of drones.
        """
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

        else:
            return
        
        # For semester project specific case
        self.update_drones_FOV(360/self.count)
        self.compute_neighborhood()

    #=======================================#
    #            Miscellaneous              #
    #=======================================#

    def print_swarm(self):
        """
        Print the members of the swarm with their positions, velocities and accelerations.
        """
        for i in range(self.count):
            print("====| DRONE {0} |=====".format(i+1))
            self.members[i].print_state()