import numpy as np

from olfati_saber import get_RB2W, get_W2B
from scipy import spatial
from controllers import PdController
from algorithms import get_viewing_dir
from helper_functions import elapsed_timer

DEFAULT_RANGE_SENSING = 2.0
DEFAULT_NB_NEIGHBORS = 3
DEFAULT_FOV = 360/10 # degrees
KP_GAIN = 2.0
KD_GAIN = 0.1
ANGULAR_RATE_LIMIT = 1.0
USE_PD_CONTROLLER = False
FOV_ASPECT_RATIO = 5/5

class Drone:
    '''
    Class to represent a drone object. 
    Contains mainly drone state (pos,vel,acc,orientation)
    '''
    def __init__(self, init_pos=[0.0]*3, init_vel=[0.0]*3,init_acc=[0.0]*3,init_angles=[0.0]*3, fov=DEFAULT_FOV):
        self.pos = np.array(init_pos, dtype=np.float64)
        self.vel = np.array(init_vel, dtype=np.float64)
        self.acc = np.array(init_acc, dtype=np.float64)
        self.angles = np.array(init_angles, dtype=np.float64) # [roll,pitch,yaw]
        self.rates = np.zeros(3, dtype=np.float64) # Angular rates
        self.mass = 1.0
        self.neighbors = list() # List of most recent neighbours (indices from swam.members list)
        self.noise = {'type': 'None'}
        self.estimated_viewing_dir = np.array([1,0,0])
        self.exact_viewing_dir = np.array([1,0,0])
        self.yaw_controller = PdController(KP_GAIN, KD_GAIN, ANGULAR_RATE_LIMIT)
        self.pitch_controller = PdController(KP_GAIN, KD_GAIN, ANGULAR_RATE_LIMIT)
        self.use_pd_controller = USE_PD_CONTROLLER
        self.viewing_error = 0.0
        self.fov = fov * np.pi / 180.0 # In radians
        self.ASPECT_RATIO = FOV_ASPECT_RATIO
        self.timing_viewing_dir = 0
        
    def get_state(self):
        return np.vstack((self.pos, self.vel, self.acc, self.angles))
    
    def get_heading(self):
        phi = self.angles[0]
        theta = self.angles[1]
        psi = self.angles[2]
        return get_RB2W(phi, theta, psi) @ np.array([1,0,0])
    
    def set_noise(self, type: str, param_dist: float, param_dir:float):
        self.noise.update({'type': type, 'param_dist': param_dist, 'param_dir': param_dir, 'param_heading': 0.0})
    
    def update(self, dt, new_acc, new_rates=np.zeros(3)):
        # COMMENTED because use directly global acceeleration update
        # To avoid numerical issues while using rotation matrices
        #self.acc = get_RB2W(self.angles[0], self.angles[1], self.angles[2]) @ new_acc.copy()
        self.acc = new_acc.copy()
        self.rates = new_rates.copy()

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

    def print_state(self):
        names = ["Pos: ", "Vel: ", "Acc: ", "Angles: "]
        state = self.get_state()
        for i in range(len(names)):
            print("{0}{1}".format(names[i], state[i]))

    def compute_neihgborhood(self, members: list["Drone"], metric, metric_data=dict()):
        index = members.index(self)
        poss_neighbors = np.array([i for i in range(len(members)) if i != index])
        sampling = metric_data.get('sampling', 1)
        noisy_poses = self._apply_sensing_noise([members[index] for index in poss_neighbors], sampling)
        match metric:
            case "Eucledian":
                sensing_range = metric_data.get('sensing_range', DEFAULT_RANGE_SENSING)
                distances = np.linalg.norm(noisy_poses - self.pos, axis=1)
                indices = np.nonzero(distances < sensing_range)[0]
                self.neighbors = [DroneNeighbor(i, distances[j], (noisy_poses[j] - self.pos) / distances[j],
                                                self._apply_noise(members[i].angles, self.noise, 'param_heading', sampling), self.pos) for i,j in zip(poss_neighbors[indices], indices)]
            case "Topological":
                nb = metric_data.get('count', DEFAULT_NB_NEIGHBORS)
                distances = np.linalg.norm(noisy_poses - self.pos, axis=1)
                indices = np.argsort(distances)[:nb]
                self.neighbors = [DroneNeighbor(i, distances[j], (noisy_poses[j] - self.pos) / distances[j],
                                                self._apply_noise(members[i].angles, self.noise, 'param_heading', sampling), self.pos) for i,j in zip(poss_neighbors[indices], indices)]
            case "Voronoi":
                points = np.concatenate((noisy_poses, self.pos.reshape(1,3)))
                indptr_neig, neighbors = spatial.Delaunay(points, qhull_options="QJ").vertex_neighbor_vertices
                self.neighbors = [DroneNeighbor(j if j<index else j+1, np.linalg.norm(points[j] - self.pos), 
                                                        (points[j] - self.pos) / np.linalg.norm(points[j] - self.pos), 
                                                        self._apply_noise(members[j].angles, self.noise, 'param_heading'), self.pos) for j in neighbors[indptr_neig[-2]:indptr_neig[-1]]]
            case "Visual LoS":
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
                
            case "TEST":
                indices = np.random.choice(poss_neighbors, np.random.randint(0, len(poss_neighbors)))
                for i in indices:
                    dist = self._apply_noise(np.linalg.norm(members[i].pos - self.pos), self.noise, 'param_dist', sampling)
                    dir = self._apply_noise(((members[i].pos - self.pos) / dist), self.noise, 'param_dir', sampling)
                    angles = self._apply_noise(members[i].angles, self.noise, 'param_heading', sampling)
                    self.neighbors.append(DroneNeighbor(i, dist, dir, angles, self.pos))
            case _:
                self.neighbors = []

        return self.neighbors
    
    def compute_viewing_dir(self, members: list["Drone"], algo_dict):
        algo = algo_dict.get('algorithm', 'None')
        nb_points = algo_dict.get('nb_points', 2)
        face_type = algo_dict.get('faces', 'adjacent')
        in_2d = algo_dict.get('in_2d', False)
        with elapsed_timer() as elapsed:
            self.estimated_viewing_dir = get_viewing_dir(self, self.neighbors, algo, n_points=nb_points, faces=face_type, in_2d=in_2d)
            self.timing_viewing_dir = elapsed()
        if algo == 'outter' and nb_points > 3:
            # Use only the ground truth without noise to compute the exact viewing direction
            self.exact_viewing_dir = np.zeros(3)
        else:
            self.exact_viewing_dir = get_viewing_dir(self, [m for m in members if m != self], algo, n_points=nb_points, faces=face_type, in_2d=in_2d)
        # self.exact_viewing_dir = get_viewing_dir(self, [m for m in members if m != self], algo, n_points=nb_points, faces=face_type, in_2d=in_2d)
        # Compute error
        self.viewing_error = np.arccos(np.clip(np.dot(self.estimated_viewing_dir, self.exact_viewing_dir),-1,1)) * 180 / np.pi

    def _apply_noise(self, value, noise, type, sampling=1):
        match noise['type']:
            case "None":
                return value
            case "Gaussian":
                return value + np.mean(np.random.normal(0, noise[type], (len(value), sampling)), axis=1)
            case "Uniform":
                return value + np.mean(np.random.uniform(-noise[type], noise[type], (len(value), sampling)))
            case _:
                return value
            
    def _apply_sensing_noise(self, neighbours, sampling):
        # Compute neighbor distances
        n_poses = np.array([n.pos for n in neighbours]) - self.pos
        dist = np.linalg.norm(n_poses, axis=1)
        noisy_dist = self._apply_noise(dist, self.noise, 'param_dist', sampling)
        dir = n_poses / dist[:, np.newaxis]
        nois_dir = np.array([self._apply_noise(d, self.noise, 'param_dir', sampling) for d in dir])
        nois_dir = nois_dir / np.linalg.norm(nois_dir, axis=1)[:, np.newaxis]
        return self.pos + nois_dir * noisy_dist[:, np.newaxis]
    
    def get_abs_pos(self):
        return self.pos
    
    def set_fov(self, fov):
        self.fov = fov * np.pi / 180.0
    

class DroneNeighbor:
    def __init__(self, drone_index:int, distance:float, dir:np.array, angles:np.array, origin:np.array):
        self.drone_index = drone_index
        self.distance = distance
        self.dir = dir
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.angles = angles
        self.origin = origin


    def get_state(self):
        pos = self.dir * self.distance
        heading = get_RB2W(self.angles[0], self.angles[1], self.angles[2]) @ np.array([1,0,0])
        return np.vstack((pos, self.vel, self.acc, heading))
    
    def get_abs_pos(self):
        return self.dir * self.distance + self.origin

