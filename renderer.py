import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import tkinter as tk
from matplotlib.animation import FuncAnimation
from swarm import Swarm
from collections import OrderedDict


PLOT_AXIS_MARGIN = 1.2

class RendererDara:
    axis_limits = np.array([[-5,5], [-5,5], [5,15]])

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
        self.configure_plot()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=True)
        self.toolbar.update()

        self._swarm_ref = swarm
        self.ani = FuncAnimation(self.fig, self.render, interval=5, cache_frame_data=False)


    def configure_plot(self):
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim(RendererDara.axis_limits[0])
        self.ax.set_ylim(RendererDara.axis_limits[1])
        self.ax.set_zlim(RendererDara.axis_limits[2])

    def center_plot_data(self):
        try:
            min_x, max_x = np.min(self.data[:,0]), np.max(self.data[:,0])
            min_y, max_y = np.min(self.data[:,1]), np.max(self.data[:,1])

            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2

            max_diff = max(abs(max_x - center_x), abs(max_y - center_y)) * PLOT_AXIS_MARGIN
            RendererDara.axis_limits[0] = np.array([center_x - max_diff,center_x + max_diff])
            RendererDara.axis_limits[1] = np.array([center_y - max_diff, center_y + max_diff])
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


class Renderer2D():
    def __init__(self, panel:tk.Frame, swarm:Swarm):
        self.master = panel
        # Single drone view definitions
        self.viewing_radius = 5
        self.selected_drone = 0
        #swarm.compute_neighborhood("TEST")
        # Initialize the figure
        self.fig = plt.figure(2, constrained_layout=True)
        gs = self.fig.add_gridspec(6,2,width_ratios=[1,2])
        self.ax = [self.fig.add_subplot(gs[0:2,0]), self.fig.add_subplot(gs[2:4,0]), self.fig.add_subplot(gs[4:6,0])]
        self.ax.append(self.fig.add_subplot(gs[1:5,1]))
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.ax_limits = np.array([[-5,5], [-5,5], [5,15]])
        # Configuring necessary plots
        self.artists = {
            "scatter_xy":self.ax[0].scatter([],[],s=40, marker='o', c='b', cmap=None, animated=True),
            "scatter_xz":self.ax[1].scatter([],[],s=40, marker='o', c='b', cmap=None, animated=True),
            "scatter_yz":self.ax[2].scatter([],[],s=40, marker='o', c='b', cmap=None, animated=True),
            "h_dashed": self.ax[3].plot([],[],'k--', animated=True, linewidth=1.5, alpha=0.6)[0],
            "v_dashed": self.ax[3].plot([],[],'k--', animated=True, linewidth=1.5, alpha=0.6)[0],
            "scatter_single": self.ax[3].scatter([],[],s=60, marker='o', c='r', cmap=None, animated=True),
            "scatter_swarm": self.ax[3].scatter([],[],s=60, marker='o', edgecolors='k', facecolors='None', cmap=None, animated=True, linestyle='dotted')
        }

        #self.configure_plots()
        self._swarm_ref = swarm
        self.ani = FuncAnimation(self.fig, init_func=self.init_plots, frames=self.frame_iter, func=self.render, interval=5, cache_frame_data=False, blit=True)

    def set_drone(self, index, viewing_radius = 5):
        self.selected_drone = index
        self.viewing_radius = viewing_radius

    def init_plots(self):
        # XY plot
        self.ax[0].set_xlabel("x")
        self.ax[0].set_ylabel("y")
        self.ax[0].set_xlim(RendererDara.axis_limits[0])
        self.ax[0].set_ylim(RendererDara.axis_limits[1])
        self.ax[0].set_title("TOP View")
        # XZ plot
        self.ax[1].set_xlabel("x")
        self.ax[1].set_ylabel("z")
        self.ax[1].set_xlim(RendererDara.axis_limits[0])
        self.ax[1].set_ylim(RendererDara.axis_limits[2])
        self.ax[1].set_title("SIDE View")
        # YZ plot
        self.ax[2].set_xlabel("y")
        self.ax[2].set_ylabel("z")
        self.ax[2].set_xlim(RendererDara.axis_limits[1])
        self.ax[2].set_ylim(RendererDara.axis_limits[2])
        self.ax[2].set_title("FRONT View")
        # Single plot
        self.ax[3].set_xlabel("x")
        self.ax[3].set_ylabel("y")
        self.ax[3].set_title("Single Drone View")

        return self.artists.values()
    
    def frame_iter(self):
        swarm_states = np.empty((1,12))
        if self._swarm_ref is not None:
            swarm_states = self._swarm_ref.get_states()
            neighbors = self._swarm_ref.members[self.selected_drone].neighbors
        yield (swarm_states,neighbors)
    
    def render(self, data):
        swarm_states = data[0]
        neighbors = data[1]
        # Plot the drones as points
        self.artists["scatter_xy"].set(offsets=swarm_states[:,:2])
        self.artists["scatter_xz"].set(offsets=swarm_states[:,(0,2)])
        self.artists["scatter_yz"].set(offsets=swarm_states[:,1:3])
        # Plot the heading as arrows
        if np.sum(np.concatenate((swarm_states[:,-3], swarm_states[:,-2]))) > 0:
            self.artists["quiver_xy"] = self.ax[0].quiver(swarm_states[:,0],swarm_states[:,1],swarm_states[:,-3],swarm_states[:,-2],width=0.005, animated=True)
        if np.sum(np.concatenate((swarm_states[:,-3], swarm_states[:,-1]))) > 0:  
            self.artists["quiver_xz"] = self.ax[1].quiver(swarm_states[:,0],swarm_states[:,2],swarm_states[:,-3],swarm_states[:,-1],width=0.005, animated=True)
        if np.sum(np.concatenate((swarm_states[:,-2], swarm_states[:,-1]))) > 0:
            self.artists["quiver_yz"] = self.ax[2].quiver(swarm_states[:,1],swarm_states[:,2],swarm_states[:,-2],swarm_states[:,-1],width=0.005, animated=True)
        # Plot single drone + neighbors
        # Apply needed rotations with respect to body frame
        t_x, t_y = swarm_states[self.selected_drone,:2]
        self.ax[3].set_xlim([-self.viewing_radius, self.viewing_radius])
        self.ax[3].set_ylim([-self.viewing_radius, self.viewing_radius])
        points = np.array([[-2*self.viewing_radius, 2*self.viewing_radius, 0, 0], 
                            [0, 0, 2*-self.viewing_radius, 2*self.viewing_radius]])
        R_2D = np.array([[swarm_states[self.selected_drone,-3], -swarm_states[self.selected_drone,-2]],
                        [swarm_states[self.selected_drone,-2], swarm_states[self.selected_drone,-3]]])
        r_points = R_2D @ points
        self.artists["h_dashed"].set_data(r_points[0,:2], r_points[1,:2])
        self.artists["v_dashed"].set_data(r_points[0,2:4], r_points[1,2:4])
        colors = ['#FF0000A0']*(len(neighbors) + 1)
        colors[-1] = '#80ff00'
        sizes = [60]*(len(neighbors) + 1)
        sizes[-1] = 150
        if len(neighbors) > 0:
            n_data = np.vstack([n.get_state()[0,0:2] for n in neighbors])
        else:
            n_data = np.array([0,0])
        self.artists["scatter_single"].set(offsets=np.vstack((n_data, [0,0])), color=colors, sizes=sizes)
        # Plot heading of estimated neighbors
        if len(neighbors) > 0:
            n_data2 = np.vstack([n.get_state()[3,0:2] for n in neighbors])
            self.artists["quiver_n"] = self.ax[3].quiver(n_data[:,0],n_data[:,1],n_data2[:,0],n_data2[:,1],width=0.005, animated=True)
        # Plot heading of the selected drone
        self.artists["arrow_single"] = self.ax[3].arrow(0, 0, swarm_states[self.selected_drone,-3], swarm_states[self.selected_drone,-2], width=0.075, head_width=0.3, head_length=0.25, fc='g', ec='g')
        # Plot swarm
        self.artists["scatter_swarm"].set(offsets=[swarm_states[i,:2] - np.array([t_x,t_y]) for i in range(self._swarm_ref.count) if i != self.selected_drone])

        return self.artists.values()
    
    def stop(self):
        self.fig.clear()
            
def main():
    root = tk.Tk()
    root.geometry("1000x800")
    root.title("2D Swarm Views")
    swarm = Swarm(10, [0,0,10,5,5,0])
    swarm.set_noise("Uniform",0.2,0.1)
    swarm.members[0].angles = np.array([0,0,np.pi/4])
    r = Renderer2D(root, swarm)
    root.mainloop()

if __name__ == "__main__":
    main()