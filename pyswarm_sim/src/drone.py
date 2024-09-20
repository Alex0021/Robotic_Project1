import numpy as np

from pyswarm_sim.src.olfati_saber import get_RB2W, get_W2B
from scipy import spatial
from pyswarm_sim.src.controllers import PdController
from pyswarm_sim.src.algorithms import get_viewing_dir
from pyswarm_sim.src.helper_functions import elapsed_timer

#========================#
# DEFAULT PARAMETERS     #
#========================#
DEFAULT_RANGE_SENSING = 2.0
DEFAULT_NB_NEIGHBORS = 3
DEFAULT_FOV = 97 # degrees
KD_GAIN = 0.1
KP_GAIN = 2.0
ANGULAR_RATE_LIMIT = 1.0
USE_PD_CONTROLLER = False
FOV_ASPECT_RATIO = 3/5

class Drone:
    '''
    Class to represent a drone object. 
    Contains mainly drone state (pos,vel,acc,orientation)
    '''
    def __init__(self, init_pos: list[float]=[0.0]*3, 
                 init_vel:list[float]=[0.0]*3,
                 init_acc: list[float]=[0.0]*3,
                 init_angles: list[float]=[0.0]*3, 
                 fov: float=DEFAULT_FOV, 
                 swarm_2d: bool=False,
                 **kwargs):
        self.pos = np.array(init_pos, dtype=np.float64)
        self.vel = np.array(init_vel, dtype=np.float64)
        self.acc = np.array(init_acc, dtype=np.float64)
        self.angles = np.array(init_angles, dtype=np.float64) # [roll,pitch,yaw]
        self.rates = np.zeros(3, dtype=np.float64) # Angular rates
        self.mass = 1.0
        self.neighbors = list() # List of most recent neighbours (indices from swam.members list)
        self.noise = {'distribution': 'None'}
        self.estimated_viewing_dir = np.array([1,0,0], dtype=np.float64)
        self.ground_truth_viewing_dir = np.array([1,0,0], dtype=np.float64)
        kp = kwargs.get('kp', KP_GAIN)
        kd = kwargs.get('kd', KD_GAIN)
        self.use_pd_controller = kwargs.get('use_pd_controller', USE_PD_CONTROLLER)
        self.yaw_controller = PdController(kp, kd, ANGULAR_RATE_LIMIT)
        self.pitch_controller = PdController(kp, kd, ANGULAR_RATE_LIMIT)
        self.viewing_error = 0.0
        self.fov = fov * np.pi / 180.0 # In radians
        self.ASPECT_RATIO = FOV_ASPECT_RATIO
        self.timing_viewing_dir = 0
        self.swarm_2d = swarm_2d

        # Drone dynamics parameters
        self.motor_thrust = np.zeros(4)

    
    def update(self, dt: float, new_acc: np.ndarray, new_rates: np.ndarray=np.zeros(3)):
        """
        Update the drone state using the new acceleration and angular rates.

        Args:
            dt (float): simulation time step
            new_acc (np.ndarray): new calculated acceleration
            new_rates (np.ndarray, optional): new calculated angular rates. Defaults to np.zeros(3).
        """
        # COMMENTED because use directly global acceeleration update
        # To avoid numerical issues while using rotation matrices
        #self.acc = get_RB2W(self.angles[0], self.angles[1], self.angles[2]) @ new_acc.copy()
        self.acc = new_acc.copy()
        self.rates = new_rates.copy()

        # If angular rates are zero (not commanded), use PD controller (if active)
        if self.use_pd_controller and np.all(self.rates == 0):
            # Apply PD controller to the angles
            heading = self.get_heading()
            viewing_pitch = np.arccos(np.linalg.norm(self.estimated_viewing_dir[:2])/np.linalg.norm(self.estimated_viewing_dir))*-np.sign(self.estimated_viewing_dir[2])
            self.rates[1] = self.pitch_controller.get_control(viewing_pitch - self.angles[1], dt)
            self.rates[2] = self.yaw_controller.get_control_from_orientation(heading[:2], self.estimated_viewing_dir[:2], dt)
        elif not self.use_pd_controller:
            self.angles[1] = np.arccos(np.linalg.norm(self.estimated_viewing_dir[:2])/np.linalg.norm(self.estimated_viewing_dir))*-np.sign(self.estimated_viewing_dir[2])
            self.angles[2] = np.arctan2(self.estimated_viewing_dir[1], self.estimated_viewing_dir[0])
        
        # Perform simple Euler forward integration
        self.vel += self.acc * dt
        self.pos += self.vel * dt
        self.angles += self.rates * dt

        # Limit angles between -pi and pi
        self.angles = np.mod(self.angles + np.pi, 2*np.pi) - np.pi

    def print_state(self):
        """
        Print the current state of the drone (position, velocity, acceleration, angles).
        """
        names = ["Pos: ", "Vel: ", "Acc: ", "Angles: "]
        state = self.get_state()
        for i in range(len(names)):
            print("{0}{1}".format(names[i], state[i]))

    def compute_neihgborhood(self, members: list["Drone"], metric: float, metric_data=dict()) -> list["DroneNeighbor"]:
        """
        Compute the neighborhood of the drone based on the metric and metric data.

        Args:
            members (list[Drone]): list of all drones in the swarm
            metric (float): metric to use for the neighborhood computation
            metric_data (dict, optional): additional data for the metric computation. Defaults to dict().
        Returns:
            list[DroneNeighbor]: list of neighbors of the drone
        """
        index = members.index(self)
        poss_neighbors = np.array([i for i in range(len(members)) if i != index])
        sampling = metric_data.get('sampling', 1)
        noisy_poses = self._apply_sensing_noise([members[index] for index in poss_neighbors], sampling)
        match metric.upper():
            case "EUCLEDIAN":
                sensing_range = metric_data.get('sensing_range', DEFAULT_RANGE_SENSING)
                distances = np.linalg.norm(noisy_poses - self.pos, axis=1)
                indices = np.nonzero(distances < sensing_range)[0]
                self.neighbors = [DroneNeighbor(i, distances[j], (noisy_poses[j] - self.pos) / distances[j],
                                                self._apply_noise(members[i].angles, self.noise, 'param_heading', sampling), self.pos) for i,j in zip(poss_neighbors[indices], indices)]
            case "TOPOLOGICAL":
                nb = metric_data.get('count', DEFAULT_NB_NEIGHBORS)
                distances = np.linalg.norm(noisy_poses - self.pos, axis=1)
                indices = np.argsort(distances)[:nb]
                self.neighbors = [DroneNeighbor(i, distances[j], (noisy_poses[j] - self.pos) / distances[j],
                                                self._apply_noise(members[i].angles, self.noise, 'param_heading', sampling), self.pos) for i,j in zip(poss_neighbors[indices], indices)]
            case "VORONOI":
                if self.swarm_2d:
                    n_dim = 2
                    q_hull_options = ""
                else:
                    n_dim = 3
                    q_hull_options = "QJ"
                points = np.concatenate((noisy_poses[:,:n_dim], self.pos[:n_dim].reshape(1,n_dim)))
                indptr_neig, neighbors = spatial.Delaunay(points, qhull_options=q_hull_options).vertex_neighbor_vertices
                self.neighbors = [DroneNeighbor(j if j<index else j+1, np.linalg.norm(noisy_poses[j] - self.pos), 
                                                        (noisy_poses[j] - self.pos) / np.linalg.norm(noisy_poses[j] - self.pos), 
                                                        self._apply_noise(members[j].angles, self.noise, 'param_heading'), self.pos) for j in neighbors[indptr_neig[-2]:indptr_neig[-1]]]
            case "VLOS":
                sensing_range = metric_data.get('sensing_range', np.inf)
                r_agent = metric_data.get('r_agent', 0.05)
                # Keep neighbors that are within the sensing range
                distances = np.linalg.norm(noisy_poses - self.pos, axis=1)
                indices = np.where(distances < sensing_range)[0]
                poss_neighbors = poss_neighbors[indices]
                distances = distances[indices]
                # Get headings and distances to all neighbors
                headings = (noisy_poses[indices] - self.pos) / distances[:, np.newaxis]
                indices = np.argsort(distances)
                self.neighbors = []
                while len(indices) > 0:
                    n_index = poss_neighbors[indices[0]]
                    self.neighbors.append(DroneNeighbor(n_index, distances[indices[0]], headings[indices[0]], 
                                                        self._apply_noise(members[n_index].angles, self.noise, 'param_heading',sampling), self.pos))
                    # Check if the neighbor is within the field of view
                    d_ij = distances[indices[0]]
                    u_ij = headings[indices[0]]
                    r_ij = r_agent/d_ij
                    to_remove = [0]
                    for i in range(1, len(indices)):
                        # Looping through all others
                        d_ik = distances[indices[i]]
                        u_ik = headings[indices[i]]
                        r_ik = r_agent/d_ik
                        # First condition (d_ij < d_ik) is already satisfied by the sorting
                        if np.linalg.norm(u_ij-u_ik) < (r_ij + r_ik):
                            to_remove.append(i)
                    indices = np.delete(indices, to_remove)
            case _:
                self.neighbors = []

        return self.neighbors
    
    def compute_viewing_dir(self, algo_dict: dict):
        """
        Compute the viewing direction of the drone based on the algorithm.

        Args:
            algo_dict (dict): algorithm parameters
        """
        algo = algo_dict.get('algorithm', 'None')
        nb_points = algo_dict.get('nb_points', 2)
        face_type = algo_dict.get('faces', 'adjacent')
        in_2d = algo_dict.get('in_2d', False)
        with elapsed_timer() as elapsed:
            self.estimated_viewing_dir = get_viewing_dir(self, self.neighbors, algo, n_points=nb_points, faces=face_type, in_2d=in_2d)
            self.timing_viewing_dir = elapsed()
        # Calculate error (in degrees)
        self.viewing_error = np.arccos(np.clip(np.dot(self.get_heading(), self.ground_truth_viewing_dir),-1,1)) * 180 / np.pi

    def _apply_noise(self, value: float, noise: dict, type: float, sampling: int=1) -> float|np.ndarray:
        """
        Apply noise to the value based on the noise parameters.

        Args:
            value (float): value to add noise to (can be single value or array)
            noise (dict): noise parameters
            type (float): type of noise to apply (distance, direction, heading)
            sampling (int, optional): Averaging over multiple noise samples. Defaults to 1.

        Returns:
            float|np.ndarray: value with noise added
        """
        match noise['distribution'].upper():
            case "NONE":
                return value
            case "GAUSSIAN":
                return np.multiply(value, np.ones_like(value) + np.mean(np.random.normal(0, noise[type], (len(value), sampling)), axis=1))
            case "UNIFORM":
                return np.multiply(value, np.ones_like(value) + np.mean(np.random.uniform(-noise[type], noise[type], (len(value), sampling))))
            case _:
                return value
            
    def _apply_sensing_noise(self, neighbours: list["Drone"], sampling: int) -> np.ndarray:
        """
        Apply noise to the distance and direction measurements of the neighbors.

        Args:
            neighbours (list[Drone]): list of neighbors
            sampling (int): number of samples to take for the noise and compute average

        Returns:
            ndarray: neighbor positions with noise
        """
        # Compute neighbor distances
        n_poses = np.array([n.pos for n in neighbours]) - self.pos
        dist = np.linalg.norm(n_poses, axis=1)
        # Add noise to distance measurements
        noisy_dist = self._apply_noise(dist, self.noise, 'param_dist', sampling)
        dir = n_poses / dist[:, np.newaxis]
        # Add noise to direction measurements (sensing cone)
        noisy_dir = np.array([self._apply_noise(d, self.noise, 'param_dir', sampling) for d in dir])
        noisy_dir = noisy_dir / np.linalg.norm(noisy_dir, axis=1)[:, np.newaxis]
        return self.pos + noisy_dir * noisy_dist[:, np.newaxis]

    #=========================#
    # GETTERS AND SETTERS     #
    #=========================#
        
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the drone.

        Returns:
            np.ndarray: [pos; vel; acc; angles]
        """
        return np.vstack((self.pos, self.vel, self.acc, self.angles))
    
    def set_pd_controller(self, active: bool, kp: float, kd: float, rate_limit: float):
        """
        Set the PD controller parameters.

        Args:
            active (bool): use pd controller or not
            kp (float): proportional gain
            kd (float): derivative gain
            rate_limit (float): rate limit for the controller
        """
        self.use_pd_controller = active
        self.yaw_controller.kp = kp
        self.yaw_controller.kd = kd
        self.pitch_controller.kp = kp
        self.pitch_controller.kd = kd
        self.yaw_controller.rate_limit = rate_limit
        self.pitch_controller.rate_limit = rate_limit
    
    def get_heading(self) -> np.ndarray:
        """
        Get the current heading of the drone.

        Returns:
            np.ndarray: heading unit vector
        """
        phi = self.angles[0]
        theta = self.angles[1]
        psi = self.angles[2]
        return get_RB2W(phi, theta, psi) @ np.array([1,0,0])
    
    def set_noise(self, distribution: str, param_dist: float, param_dir:float, param_heading: float=0.0):
        """
        Set the noise parameters for the drone.

        Args:
            distribution (str): distribution of the noise (None, Gaussian, Uniform)
            param_dist (float): distance sensing noise 
            param_dir (float): sensing cone noise
            param_heading (float): heading noise
        """
        self.noise.update({'distribution': distribution, 'param_dist': param_dist, 'param_dir': param_dir, 'param_heading': param_heading})

    def get_abs_pos(self) -> np.ndarray:
        """
        Get the absolute position of the drone.

        Returns:
            np.ndarray: absolute position
        """
        return self.pos
    
    def set_fov(self, fov: float):
        """
        Set the field of view of the drone.

        Args:
            fov (float): field of view in degrees
        """
        self.fov = fov * np.pi / 180.0
    

class DroneNeighbor:
    """
    Class to represent a detected drone neighbor object.

    Keeps track of the drone index, distance, direction, angles and origin of the neighbor.

    """
    def __init__(self, drone_index:int, distance:float, dir:np.array, angles:np.array, origin:np.array):
        self.drone_index = drone_index
        self.distance = distance
        self.dir = dir
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.angles = angles
        self.origin = origin

    def get_state(self) -> np.ndarray:
        """
        Get the current state of the drone neighbor.

        Returns:
            np.ndarray: [pos; vel; acc; heading]
        """
        pos = self.dir * self.distance
        heading = get_RB2W(self.angles[0], self.angles[1], self.angles[2]) @ np.array([1,0,0])
        return np.vstack((pos, self.vel, self.acc, heading))
    
    def get_abs_pos(self) -> np.ndarray:
        """
        Get the absolute position of the drone neighbor.

        Returns:
            np.ndarray: absolute position (including noise)
        """
        return self.dir * self.distance + self.origin

