import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class Obstacle:
    """
    This base class contains necessary information for an obstacle implementation.
    """
    def __init__(self, center: np.ndarray, color: tuple=(1.0, 0.0, 0.0)):
        # CONSTANTS
        self.MESH_RESOLUTION = 50
        self.center = np.array(center)
        self.color = color

    def __class__(self) -> str:
        return "Obstacle"

    @abstractmethod
    def render(self, ax) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method renders the obstacle in the simulation environment.

        Args:
            ax (Axes object): axes object used for rendering

        Returns:
            tuple: grid meshpoints for rendering (X, Y, Z)
        """
        pass

    def to_dict(self) -> dict:
        return {"center": self.center, "color": self.color}


class Cylinder(Obstacle):

    """
    This class defines a cylinder obstacle.
    """
    def __init__(self, center: np.ndarray, radius: float, height: float, color: tuple=(0.0, 0.0, 1.0), **kwargs):
        super().__init__(center, color)
        self.radius = radius
        self.height = height

        # Rendering parameters
        self.alpha = 0.2
        self.edgecolor = kwargs.get('edgecolor', 'k')
        self.rcount = kwargs.get('rcount', 5)
        self.ccount = kwargs.get('ccount', 10)

    #===========================================================================
    # Rendering inspired from this forum thread: 
    # https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
    #===========================================================================
    def render(self, ax):
        z_lim = (self.center[2] - self.height/2, self.center[2] + self.height/2)
        z = np.linspace(z_lim[0], z_lim[1], self.MESH_RESOLUTION)
        theta = np.linspace(0, 2*np.pi, self.MESH_RESOLUTION)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = self.radius*np.cos(theta_grid) + self.center[0]
        y_grid = self.radius*np.sin(theta_grid) + self.center[1]
        return ax.plot_surface(x_grid, y_grid, z_grid, color=self.color, alpha=self.alpha, edgecolor=self.edgecolor, 
                               rcount=self.rcount, ccount=self.ccount)

    def __str__(self) -> str:
        return "Cylinder: center={}, radius={}, height={}".format(self.center, self.radius, self.height)

    def __class__(self) -> str:
        return "Cylinder"
    
    def to_dict(self) -> dict:
        return {"type": self.__class__(), "center": self.center.tolist(), "radius": self.radius, "height": self.height, "color": self.color}
