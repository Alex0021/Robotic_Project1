"""
    This file contains the controllers for the application.
"""

import numpy as np

class PdController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.last_error = None

    def get_control(self, error, dt):
        # First initialization
        if self.last_error is None:
            self.last_error = error
            return self.kp * error
        control = self.kp * error - self.kd * (error - self.last_error) / dt
        self.last_error = error
        return control
    
    def get_control_from_orientation(self, current, desired, dt):
        """
        """
        angle = np.arctan2(current[0]*desired[1]-current[1]*desired[0], np.dot(current, desired))
        return self.get_control(angle, dt)