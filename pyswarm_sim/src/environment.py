import numpy as np
from pyswarm_sim.src.obstacles import Cylinder, Obstacle
from matplotlib.artist import Artist

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
        self.current_selected_obs = None
        self.obs_selected_mouse = False

        # Rendering parameters
        self.target_color = kwargs.get('target_color', 'r')
        self.target_marker = kwargs.get('target_marker', 'x')
        self.target_size = kwargs.get('target_size', 20)

    def render(self, ax) -> dict["Artist"]:
        artists = dict()
        # Render obstacles
        for idx, obstacle in enumerate(self.obstacles):
            artists[f'obs_{idx}'] = obstacle.render(ax)

        # Render target (migration point) if it exists
        if self.target is not None:
            ax.plot(self.target[0], self.target[1], self.target[2], marker=self.target_marker, color=self.target_color, markersize=self.target_size)

        return artists

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
        new_obs = None
        match type.upper():
            case 'CYLINDER':
                new_obs = Cylinder(**kwargs)
                self.obstacles.append(new_obs)
            case _:
                raise ValueError("Invalid obstacle type: {0}".format(type))
            
        if 'selected' in kwargs and kwargs['selected']:
            self.select_obstacle(new_obs)
        return new_obs
            
    def remove_obstacle(self, index: int|np.ndarray):
        if isinstance(index, int):
            if index < 0 or index >= len(self.obstacles):
                raise ValueError("Invalid index: {0}".format(index))
            self.obstacles.pop(index)
        else:
            to_keep = np.ones(len(self.obstacles), dtype=bool)
            to_keep[index] = False
            self.obstacles = np.array(self.obstacles)[to_keep].tolist()

    def select_obstacle(self, obs: int|Obstacle, with_mouse: bool = False):
        if self.current_selected_obs is not None:
            self.obstacles[self.current_selected_obs].selected = False
        if isinstance(obs, Obstacle):
            if obs not in self.obstacles:
                raise ValueError("Obstacle not found in the environment")
            self.current_selected_obs = self.obstacles.index(obs)
            obs.selected = True
            self.obs_selected_mouse = with_mouse
        elif isinstance(obs, int):
            index = obs
            if index < 0 or index >= len(self.obstacles):
                raise ValueError("Invalid index: {0}".format(index))
            self.current_selected_obs = index   
            self.obstacles[index].selected = True       
            self.obs_selected_mouse = with_mouse

    def deselect_obstacle(self):
        self.current_selected_obs = None
        self.obs_selected_mouse = False
        for obs in self.obstacles:
            obs.selected = False

    def get_selected_obstacle(self) -> Obstacle:
        if self.current_selected_obs is None:
            return None
        return self.obstacles[self.current_selected_obs]

    