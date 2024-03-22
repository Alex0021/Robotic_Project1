import numpy as np

from olfati_saber import get_RB2W, get_W2B
from scipy import spatial

DEFAULT_RANGE_SENSING = 2.0
DEFAULT_NB_NEIGHBORS = 3

class Drone:
    '''
    Class to represent a drone object. 
    Contains mainly drone state (pos,vel,acc,orientation)
    '''
    def __init__(self, init_pos=[0.0]*3, init_vel=[0.0]*3,init_acc=[0.0]*3,init_angles=[0.0]*3):
        self.pos = np.array(init_pos, dtype=np.float64)
        self.vel = np.array(init_vel, dtype=np.float64)
        self.acc = np.array(init_acc, dtype=np.float64)
        self.angles = np.array(init_angles, dtype=np.float64) # [roll,pitch,yaw]
        self.rates = np.zeros(3, dtype=np.float64) # Angular rates
        self.mass = 1.0
        self.neighbors = list() # List of most recent neighbours (indices from swam.members list)
        
    def get_state(self):
        return np.vstack((self.pos, self.vel, self.acc, self.angles))
    
    def get_heading(self):
        phi = self.angles[0]
        theta = self.angles[1]
        psi = self.angles[2]
        return get_RB2W(phi, theta, psi) @ np.array([1,0,0])
    
    def update(self, dt, new_acc, new_rates=np.zeros(3)):
        self.acc = get_RB2W(self.angles[0], self.angles[1], self.angles[2]) @ new_acc
        self.rates = new_rates
        
        # Perform simple Euler forward integration
        self.vel += self.acc * dt
        self.pos += self.vel * dt
        self.angles += self.rates * dt


    def print_state(self):
        names = ["Pos: ", "Vel: ", "Acc: ", "Angles: "]
        state = self.get_state()
        for i in range(len(names)):
            print("{0}{1}".format(names[i], state[i]))

    def compute_neihgborhood(self, members: list["Drone"], metric, noise, metric_data=dict()):
        index = members.index(self)
        poss_neighbors = np.array([i for i in range(len(members)) if i != index])
        match metric:
            case "Eucledian":
                sensing_range = metric_data.get('sensing_range', DEFAULT_RANGE_SENSING)
                distances = np.array([np.linalg.norm(members[i].pos - self.pos) for i in poss_neighbors])
                indices = np.nonzero(distances < sensing_range)
                self.neighbors = [DroneNeighbor(i, distances[j], (members[i].pos - self.pos) / distances[j], members[i].angles) for i,j in zip(poss_neighbors[indices], indices)]
            case "Topological":
                nb = metric_data.get('nb_neighbors', DEFAULT_NB_NEIGHBORS)
                distances = np.array([np.linalg.norm(members[i].pos - self.pos) for i in poss_neighbors])
                indices = np.argsort(distances)[:nb]
                self.neighbors = [DroneNeighbor(i, self._apply_noise(distances[j], noise, 'param_pos'), 
                                                self._apply_noise((members[i].pos - self.pos) / distances[j], noise, 'param_pos'), 
                                                self._apply_noise(members[i].angles, noise, 'param_heading')) for i,j in zip(poss_neighbors[indices], indices)]
            case "Voronoi":
                # This should be called once on one of the drone only
                points = np.array([members[i].pos for i in range(len(members))])
                indptr_neig, neighbors = spatial.Delaunay(points, qhull_options="QJ").vertex_neighbor_vertices
                for i in range(len(members)):
                    members[i].neighbors = [DroneNeighbor(j, self._apply_noise(np.linalg.norm(members[j].pos - members[i].pos),noise, 'param_pos'), 
                                                          self._apply_noise((members[j].pos - members[i].pos) / np.linalg.norm(members[j].pos - members[i].pos), noise, 'param_pos'), 
                                                          self._apply_noise(members[j].angles, noise, 'param_heading')) for j in neighbors[indptr_neig[i]:indptr_neig[i+1]]]
            case "Visual LoS":
                sensing_range = metric_data.get('sensing_range', np.inf)
                r_agent = metric_data.get('r_agent', 0.05)
                # Keep neighbors that are within the sensing range
                distances = np.array([np.linalg.norm(members[i].pos - self.pos) for i in poss_neighbors])
                distances = distances[distances < sensing_range]
                poss_neighbors = poss_neighbors[distances < sensing_range]
                # Get headings and distances to all neighbors
                headings = np.array([(members[i].pos - self.pos)/distances[j] for i,j in zip(poss_neighbors, range(len(poss_neighbors)))])
                indices = np.argsort(distances)
                self.neighbors = []
                while len(indices) > 0:
                    n_index = poss_neighbors[indices[0]]
                    self.neighbors.append(DroneNeighbor(n_index, self._apply_noise(distances[indices[0]], noise, 'param_pos'), 
                                                        self._apply_noise(headings[indices[0]], noise, 'param_pos'), 
                                                        self._apply_noise(members[n_index].angles), noise, 'param_heading'))
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
                    dist = self._apply_noise(np.linalg.norm(members[i].pos - self.pos), noise, 'param_pos')
                    dir = self._apply_noise(((members[i].pos - self.pos) / dist), noise, 'param_pos')
                    angles = self._apply_noise(members[i].angles, noise, 'param_heading')
                    self.neighbors.append(DroneNeighbor(i, dist, dir, angles))
            case _:
                self.neighbors = []

        return self.neighbors
    
    def _apply_noise(self, value, noise, type):
        match noise['type']:
            case "None":
                return value
            case "Gaussian":
                return value + np.random.normal(0, noise[type])
            case "Uniform":
                return value + np.random.uniform(-noise[type], noise[type])
            case _:
                return value
    

class DroneNeighbor:
    def __init__(self, drone_index:int, distance:float, dir:np.array, angles:np.array):
        self.drone_index = drone_index
        self.distance = distance
        self.dir = dir
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.angles = angles


    def get_state(self):
        pos = self.dir * self.distance
        heading = get_RB2W(self.angles[0], self.angles[1], self.angles[2]) @ np.array([1,0,0])
        return np.vstack((pos, self.vel, self.acc, heading))

