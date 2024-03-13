import numpy as np

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
        return Drone.get_RB2W(phi, theta, psi) @ np.array([1,0,0])
    
    def update(self, dt, new_acc, new_rates=np.zeros(3)):
        self.acc = new_acc
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

    def compute_neihgborhood(self, members: list["Drone"], metric, noise):
        index = members.index(self)
        poss_neighbors = [i for i in range(len(members)-1) if i != index]
        match metric:
            case "Eucledian":
                pass
            case "Topological":
                pass
            case "Voronoi":
                pass
            case "Visual LoS":
                pass
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

    def get_RB2W(phi, theta, psi):
        R_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0,0,1]])
        R_theta = np.array([[np.cos(theta), 0,np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]])
        R_phi = [[1,0,0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        return R_psi @ R_theta @ R_phi
    
    def get_W2B(phi, theta, psi):
        R_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0,0,1]])
        R_theta = np.array([[np.cos(theta), 0,np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]])
        R_phi = [[1,0,0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        return np.transpose(R_phi @ R_theta @ R_psi)
    

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
        heading = Drone.get_RB2W(self.angles[0], self.angles[1], self.angles[2]) @ np.array([1,0,0])
        return np.vstack((pos, self.vel, self.acc, heading))

