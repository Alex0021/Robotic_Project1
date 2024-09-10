import numpy as np
from abc import ABC, abstractmethod

class Obstacle:
    """
    This base class contains necessary information for an obstacle implementation.
    """
    def __init__(self, center: np.ndarray, color: tuple=(255, 0, 0)):
        self.center = np.array(center)
        self.color = color

    @abstractmethod
    def render(self):
        pass


class Cylinder(Obstacle):
    """
    This class defines a cylinder obstacle.
    """
    def __init__(self, center: np.ndarray, radius: float, height: float, color: tuple=(255, 0, 0), **kwargs):
        super().__init__(center, color)
        self.radius = radius
        self.height = height

    def render(self):
        pass

    def __str__(self) -> str:
        return "Cylinder: center={}, radius={}, height={}".format(self.center, self.radius, self.height)

