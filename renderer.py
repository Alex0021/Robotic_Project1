import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import tkinter as tk
from matplotlib.animation import FuncAnimation
from swarm import Swarm

PLOT_AXIS_MARGIN = 1.2

class Renderer():
    def __init__(self, panel:tk.Frame, swarm:Swarm):
        self.master = panel
        # Initialize the figure
        self.fig = plt.figure(1)
        self.fig.suptitle("Live Swarm View")
        self.fig.subplots_adjust(top=1.1, bottom=-0.1, left=-0.1, right=1.1)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(18,-170,0)
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.ax_limits = np.array([[0,10], [-10,10], [5,15]])
        self.configure_plot()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=True)
        self.toolbar.update()

        # Initialize a data queue for safe inter-thread update
        self._swarm_ref = swarm
        self.ani = FuncAnimation(self.fig, self.render, interval=5)


    def configure_plot(self):
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim(self.ax_limits[0])
        self.ax.set_ylim(self.ax_limits[1])
        self.ax.set_zlim(self.ax_limits[2])

    def center_plot_data(self):
        try:
            min_x, max_x = np.min(self.data[:,0]), np.max(self.data[:,0])
            min_y, max_y = np.min(self.data[:,1]), np.max(self.data[:,1])

            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2

            max_diff = max(abs(max_x - center_x), abs(max_y - center_y)) * PLOT_AXIS_MARGIN
            self.ax_limits[0] = np.array([center_x - max_diff,center_x + max_diff])
            self.ax_limits[1] = np.array([center_y - max_diff, center_y + max_diff])
        except:
            return

    
    def render(self, i):
        if self._swarm_ref is not None:
            self.data = self._swarm_ref.get_states()
            self.ax.clear()
            self.configure_plot()
            # Plot the drones as points
            self.ax.scatter(self.data[:,0],self.data[:,1],self.data[:,2],s=40, marker='o', c='b', cmap=None)
            # Plot the heading as arrows
            self.ax.quiver(self.data[:,0],self.data[:,1],self.data[:,2], self.data[:,-3], self.data[:,-2], self.data[:,-1], length=0.25, normalize=True)
            # Plot the migration point
            if self._swarm_ref.migration_point is not None:
                self.ax.plot(self._swarm_ref.migration_point[0], self._swarm_ref.migration_point[1], self._swarm_ref.migration_point[2],'r', marker='x', markersize=20)

    def stop(self):
        self.ani.event_source.stop()

    def start(self):
        self.ani.event_source.start()
    
    def reset(self):
        self.ax.clear()
        self.configure_plot()
        self.canvas.draw()
        self.ax.view_init(18,-170,0)