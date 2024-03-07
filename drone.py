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
        
    def get_state(self):
        return np.vstack((self.pos, self.vel, self.acc, self.angles))
    
    def get_heading(self):
        phi = self.angles[0]
        theta = self.angles[1]
        psi = self.angles[2]
        R_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0,0,1]])
        R_theta = np.array([[np.cos(theta), 0,np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]])
        R_phi = [[1,0,0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        return R_psi @ R_theta @ R_phi @ np.array([1,0,0])
    
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