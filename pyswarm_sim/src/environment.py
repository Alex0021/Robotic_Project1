import numpy as np
from pyswarm_sim.src.obstacles import Cylinder

class Environment:
    def __init__(self, obstacles: list, target: np.ndarray = None, **kwargs):
        self.obstacles = []
        for obs in obstacles:
            obs_type = obs.get('type', 'unknown')
            match obs_type.upper():
                case 'CYLINDER':
                    self.obstacles.append(Cylinder(**obs))
                case _:
                    raise ValueError("Invalid obstacle type: {0}".format(obs_type))

        self.set_target(target)

        # Rendering parameters
        self.target_color = kwargs.get('target_color', 'r')
        self.target_marker = kwargs.get('target_marker', 'x')
        self.target_size = kwargs.get('target_size', 20)

    def render(self, ax):
        # Render obstacles
        for obstacle in self.obstacles:
            obstacle.render(ax)

        # Render target (migration point) if it exists
        if self.target is not None:
            ax.plot(self.target[0], self.target[1], self.target[2], marker=self.target_marker, color=self.target_color, markersize=self.target_size)

    def set_target(self, target: np.ndarray|str):
        if isinstance(target, str):
            if len(target) == 0:
                self.target = None
                return
            try:
                self.target = np.asarray(target.split(';'), dtype=float)
            except ValueError:
                self.target = None
        else:
            self.target = target

    def add_obstacle(self, type: str, **kwargs):
        match type.upper():
            case 'CYLINDER':
                self.obstacles.append(Cylinder(**kwargs))
            case _:
                raise ValueError("Invalid obstacle type: {0}".format(type))
            
    def remove_obstacle(self, index: int|np.ndarray):
        if isinstance(index, int):
            if index < 0 or index >= len(self.obstacles):
                raise ValueError("Invalid index: {0}".format(index))
            self.obstacles.pop(index)
        else:
            to_keep = np.ones(len(self.obstacles), dtype=bool)
            to_keep[index] = False
            self.obstacles = np.array(self.obstacles)[to_keep].tolist()

    