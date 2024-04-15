"""
    This file contains the controllers for the application.
"""

import numpy as np

class PdController:
    def __init__(self, kp, kd, rate_limit=np.inf):
        self.kp = kp
        self.kd = kd
        self.last_error = None
        self.rate_limit = rate_limit

    def get_control(self, error, dt):
        # First initialization
        if self.last_error is None:
            self.last_error = error
            return self.kp * error
        control = self.kp * error + self.kd * (error - self.last_error) / dt
        self.last_error = error
        return np.clip(control, -self.rate_limit, self.rate_limit)
    
    def get_control_from_orientation(self, current, desired, dt):
        """
        """
        angle = np.arctan2(current[0]*desired[1]-current[1]*desired[0], np.dot(current, desired))
        return self.get_control(angle, dt)