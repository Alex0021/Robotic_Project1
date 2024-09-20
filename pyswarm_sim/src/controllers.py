"""
    This file contains the controllers for the application.
"""

import numpy as np

class PdController:
    """
    A simple PD controller.
    """
    def __init__(self, kp: float, kd: float, rate_limit: float=np.inf):
        self.kp = kp
        self.kd = kd
        self.last_error = None
        self.rate_limit = rate_limit

    def get_control(self, error: float, dt: float) -> float:
        """
        Get the control signal from the error and the time step.

        Args:
            error (float): Error signal
            dt (float): Simulation time step

        Returns:
            float: Control signal
        """
        # First initialization
        if self.last_error is None:
            self.last_error = error
            return self.kp * error
        control = self.kp * error + self.kd * (error - self.last_error) / dt
        self.last_error = error
        return np.clip(control, -self.rate_limit, self.rate_limit)
    
    def get_control_from_orientation(self, current: float, desired: float, dt: float) -> float:
        """
        Get the control signal from the current and desired orientation.

        Args:
            current (float): current orientation
            desired (float): desired orientation
            dt (float): simulation time step

        Returns:
            float: control signal
        """
        angle = np.arctan2(current[0]*desired[1]-current[1]*desired[0], np.dot(current, desired))
        return self.get_control(angle, dt)