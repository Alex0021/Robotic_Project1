import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import tkinter as tk
from matplotlib.animation import FuncAnimation
from pyswarm_sim.src.swarm import Swarm
from pyswarm_sim.src.environment import Environment
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
from collections import defaultdict

# Set the default keymap to close the window to ctrl+w
plt.rcParams['keymap.quit'] = 'ctrl+w'
plt.rcParams['keymap.save'] = 'ctrl+s'

PLOT_AXIS_MARGIN = 1.2
DEBUG_VLOS = False # Draw lines to all neighbors

class RendererData:
    """
        Class used to share data between the 3D and 2D renderers
    """
    axis_limits = np.array([[-5,5], [-5,5], [0,10]], dtype=np.float64)
    viewing_radius = 5
    axis_limits_single = np.array([[-viewing_radius,viewing_radius], [-viewing_radius,viewing_radius]], dtype=np.float64)

class Renderer():
    def __init__(self, panel:tk.Frame, swarm:Swarm, env: Environment):
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
        self.canvas.mpl_connect('pick_event', self.onpick)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_moved)
        self.configure_plot()
        self.artists_dict = defaultdict(lambda: None)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=True)
        self.toolbar.update()
        # Should be temporary
        # Needs to find a correct way to extract coordinates from mouse mouvements
        # event.xdata, event.ydata are not correct with 3D Axes
        self.coord_lbl = self.toolbar.children['!label2']

        self._env = env

        self._swarm_ref = swarm
        self.show_neighbors = True
        self.ani = FuncAnimation(self.fig, self.render, interval=5, cache_frame_data=False)


    def configure_plot(self):
        """
        Initial configuration of the plot
        """
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_xlim(RendererData.axis_limits[0])
        self.ax.set_ylim(RendererData.axis_limits[1])
        self.ax.set_zlim(RendererData.axis_limits[2])

    def center_plot_data(self):
        """
        Center the plot data in the middle of the axis limits to include all the data in the plot
        """
        try:
            min_x, max_x = np.min(self.data[:,0]), np.max(self.data[:,0])
            min_y, max_y = np.min(self.data[:,1]), np.max(self.data[:,1])
            min_z, max_z = np.min(self.data[:,2]), np.max(self.data[:,2])

            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            center_z = (max_z + min_z) / 2

            max_diff = max(abs(max_x - center_x), abs(max_y - center_y), abs(max_z - center_z)) * PLOT_AXIS_MARGIN
            RendererData.axis_limits[0] = [center_x - max_diff,center_x + max_diff]
            RendererData.axis_limits[1] = [center_y - max_diff, center_y + max_diff]
            RendererData.axis_limits[2] = [center_z - max_diff, center_z + max_diff]
        except:
            return
        
    def reset_view(self):
        """
        Reset to original view
        """
        RendererData.axis_limits = np.array([[-5,5], [-5,5], [0,10]], dtype=np.float64)

    def disable_rendering(self):
        """
        Disable the rendering of the plot
        """
        self.ani.event_source.stop()
        self.canvas.get_tk_widget().pack_forget()
        self.toolbar.pack_forget()

    
    def render(self, i: int):
        """
        Render the 3D view of the swarm

        Args:
            i (int): frame index
        """
        self.ax.clear()
        self.configure_plot()
        if self._swarm_ref is not None:
            try:
                self.data = self._swarm_ref.get_states().copy()
                # Plot the drones as points
                colors = np.array(['#0000FFFF']*self._swarm_ref.count)
                colors[self._swarm_ref.selected_drone] = '#00ff00ff'
                if self.show_neighbors:
                    neighbors_id = [n.drone_index for n in self._swarm_ref.members[self._swarm_ref.selected_drone].neighbors]
                    if len(neighbors_id) > 0:
                        colors[neighbors_id] = '#ff0000ff'
                #sizes = [self._get_size_in_points(self._swarm_ref.member_size)]*self._swarm_ref.count
                sizes = [100]*self._swarm_ref.count
                self.artists_dict['drones'] = self.ax.scatter(self.data[:,0],self.data[:,1],self.data[:,2],s=sizes, marker='o', color=colors.tolist(), cmap=None, picker=True, depthshade=True)
                # Plot the heading as arrows
                self.ax.quiver(self.data[:,0],self.data[:,1],self.data[:,2], self.data[:,-3], self.data[:,-2], self.data[:,-1], length=0.25, normalize=True)
                # Plot the desired viewing direction
                id = self._swarm_ref.selected_drone
                self.ax.quiver(self.data[id,0],self.data[id,1],self.data[id,2], self._swarm_ref.members[id].ground_truth_viewing_dir[0], 
                               self._swarm_ref.members[id].ground_truth_viewing_dir[1], self._swarm_ref.members[id].ground_truth_viewing_dir[2], 
                               color='#55FF00FF', length=0.5, normalize=True)
                # VLOS DEBUG
                if DEBUG_VLOS:
                    index = self._swarm_ref.selected_drone
                    for i in range(self._swarm_ref.count):
                        self.ax.plot([self.data[i,0], self.data[index,0]], [self.data[i,1], self.data[index,1]], [self.data[i,2], self.data[index,2]], 'k--', alpha=0.5)
            except Exception as e:
                print(e)

        # RENDER ENV
        if self._env is not None:
            self.artists_dict.update(self._env.render(self.ax))


    def stop(self):
        """
        Stop the animation
        """
        self.ani.event_source.stop()

    def start(self):
        """
        Start the animation
        """
        self.ani.event_source.start()
    
    def reset(self):
        """
        Reset the plot to the initial state
        """
        self.ax.clear()
        self.configure_plot()
        self.canvas.draw()

    def onpick(self, event: tk.Event):
        """
        Callback function for the pick event when selecting a point

        Args:
            event (tk.Event): data of the event

        """
        
        artist = event.artist

        if artist == self.artists_dict['drones']:
            index = event.ind
            if index is None:
                return
            print("Selected drone: {0}".format(index[0]))
            self._swarm_ref.selected_drone = index[0]
            # Update main app noise panel
            try:
                self.master.master.drone_selection_changed()
            except:
                pass

        else:
            for k,v in self.artists_dict.items():
                if k.startswith('obs'):
                    if v == artist:
                        obs_idx = int(k.split('_')[1])
                        print(f"Selected obstacle: {obs_idx}")
                        self._env.select_obstacle(obs_idx, with_mouse=True)
                        self.master.master.obstacle_clicked_callback()
                        self.btn_press_cid = self.canvas.mpl_connect('button_press_event', self.on_mouse_pressed)
                        break

    def on_mouse_pressed(self, event: tk.Event):
        if event.button == MouseButton.LEFT:
            # Check if an obstacle is already selected
            if self._env.obs_selected_mouse:
                self._env.obs_selected_mouse = False
                self.master.master.obstacle_moved_callback()
                self.canvas.mpl_disconnect(self.btn_press_cid)

    def on_mouse_moved(self, event: tk.Event):
        if not (event.inaxes == self.ax):
            return
        if self._env.obs_selected_mouse:
            # Move the obstacle
            # self._env.move_obstacle(np.array([event.xdata, event.ydata, 0]))
            # Temporary work around to get coordinates using toolbar label
            try:
                x, y, _ = [float(s.split('=')[-1]) for s in self.coord_lbl['text'].replace(u'\u2212', '-').split(',')]
                obs = self._env.get_selected_obstacle()
                obs.center[0] = x
                obs.center[1] = y
            except:
                return


class Renderer2D():
    """
        Class used to render the 2D views of the swarm

        !! Need improvements to be less computationally expensive and more interactive
    """
    def __init__(self, panel:tk.Frame, swarm:Swarm):
        self.master = panel
        # Single drone view definitions
        self.viewing_radius = 5

        # Initialize the figure
        self.fig = plt.figure(2, constrained_layout=True)
        gs = self.fig.add_gridspec(6,2,width_ratios=[1,2])
        self.ax = [self.fig.add_subplot(gs[0:2,0]), self.fig.add_subplot(gs[2:4,0]), self.fig.add_subplot(gs[4:6,0])]
        self.ax.append(self.fig.add_subplot(gs[1:5,1]))
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('pick_event', self.onpick)
        self.ax_limits = np.array([[-5,5], [-5,5], [5,15]])
        # Add slider
        ax_viewing = self.fig.add_axes([0.55, 0.07, 0.35, 0.05])
        self.freq_slider = Slider(
            ax=ax_viewing,
            label='Viewing Radius',
            valmin=1,
            valmax=10,
            valinit=5,
            valstep=0.1
        )
        self.freq_slider.on_changed(self.update_viewing_radius)
        # Configuring necessary plots
        self.ax[3].grid(True)
        self.ax[3].set_aspect(1)
        self.artists = {
            "scatter_xy":self.ax[0].scatter([],[],s=40, marker='o', c='b', cmap=None, animated=True, picker=True),
            "scatter_xz":self.ax[1].scatter([],[],s=40, marker='o', c='b', cmap=None, animated=True, picker=True),
            "scatter_yz":self.ax[2].scatter([],[],s=40, marker='o', c='b', cmap=None, animated=True, picker=True),
            "h_dashed": self.ax[3].plot([],[],'k--', animated=True, linewidth=1.5, alpha=0.6)[0],
            "v_dashed": self.ax[3].plot([],[],'k--', animated=True, linewidth=1.5, alpha=0.6)[0],
            "scatter_single": self.ax[3].scatter([],[],s=60, marker='o', c='r', cmap=None, animated=True),
            "scatter_swarm": self.ax[3].scatter([],[],s=60, marker='o', edgecolors='k', facecolors='None', cmap=None, animated=True, linestyle='dotted',picker=True)
        }

        #self.configure_plots()
        self._swarm_ref = swarm
        self.ani = FuncAnimation(self.fig, init_func=self.init_plots, frames=self.frame_iter, func=self.render, interval=5, cache_frame_data=False, blit=True)

    def update_viewing_radius(self, val: float):
        """
        Change the axis limits of the single drone view

        Args:
            val (float): New viewing radius
        """
        self.viewing_radius = val
        RendererData.axis_limits_single = np.array([[-self.viewing_radius,self.viewing_radius], [-self.viewing_radius,self.viewing_radius]])
        self.init_plots()

    def init_plots(self):
        # XY plot
        #self.ax[0].clear()
        self.ax[0].set_xlabel("x")
        self.ax[0].set_ylabel("y")
        self.ax[0].set_xlim(RendererData.axis_limits[0])
        self.ax[0].set_ylim(RendererData.axis_limits[1])
        self.ax[0].set_title("TOP View")
        # XZ plot
        self.ax[1].set_xlabel("x")
        self.ax[1].set_ylabel("z")
        self.ax[1].set_xlim(RendererData.axis_limits[0])
        self.ax[1].set_ylim(RendererData.axis_limits[2])
        self.ax[1].set_title("SIDE View")
        # YZ plot
        self.ax[2].set_xlabel("y")
        self.ax[2].set_ylabel("z")
        self.ax[2].set_xlim(RendererData.axis_limits[1])
        self.ax[2].set_ylim(RendererData.axis_limits[2])
        self.ax[2].set_title("FRONT View")
        # Single plot
        self.ax[3].set_xlabel("x")
        self.ax[3].set_ylabel("y")
        self.ax[3].set_xlim(RendererData.axis_limits_single[0])
        self.ax[3].set_ylim(RendererData.axis_limits_single[1])
        self.ax[3].set_title("Single Drone View")

        return self.artists.values()
    
    def frame_iter(self):
        swarm_states = np.empty((1,12))
        if self._swarm_ref is not None:
            swarm_states = self._swarm_ref.get_states()
            selected_drone = self._swarm_ref.selected_drone
            neighbors = self._swarm_ref.members[selected_drone].neighbors
        yield (swarm_states,neighbors, selected_drone)
    
    def render(self, data: tuple) -> list:
        """
        Render the 2D views of the swarm

        Args:
            data (tuple): Yielded data from the frame_iter function

        Returns:
            list: List of artists to be rendered
        """
        swarm_states = data[0]
        neighbors = data[1]
        selected_drone = data[2]
        try:
            # Plot the drones as points
            colors = ['#0000FFFF']*self._swarm_ref.count
            colors[selected_drone] = '#80ff00'
            for n in neighbors:
                    colors[n.drone_index] = '#ff0000ff'
            # Compute equivalent sizes
            sizes = []
            for i in range(len(self.ax)):
                ppd=72./self.ax[i].figure.dpi
                trans = self.ax[i].transData.transform
                sizes.append(np.mean((trans((2*self._swarm_ref.member_size,2*self._swarm_ref.member_size))-trans((0,0)))*ppd)**2)

            self.artists["scatter_xy"].set(offsets=swarm_states[:,:2], color=colors, sizes=sizes[0]*np.ones(self._swarm_ref.count))
            self.artists["scatter_xz"].set(offsets=swarm_states[:,(0,2)], color=colors, sizes=sizes[1]*np.ones(self._swarm_ref.count))
            self.artists["scatter_yz"].set(offsets=swarm_states[:,1:3], color=colors, sizes=sizes[2]*np.ones(self._swarm_ref.count))
            # Plot the heading as arrows
            if np.sum(np.abs(np.concatenate((swarm_states[:,-3], swarm_states[:,-2])))) > 0:
                self.artists["quiver_xy"] = self.ax[0].quiver(swarm_states[:,0],swarm_states[:,1],swarm_states[:,-3],swarm_states[:,-2],width=0.005, animated=True)
            if np.sum(np.abs(np.concatenate((swarm_states[:,-3], swarm_states[:,-1])))) > 0:  
                self.artists["quiver_xz"] = self.ax[1].quiver(swarm_states[:,0],swarm_states[:,2],swarm_states[:,-3],swarm_states[:,-1],width=0.005, animated=True)
            if np.sum(np.abs(np.concatenate((swarm_states[:,-2], swarm_states[:,-1])))) > 0:
                self.artists["quiver_yz"] = self.ax[2].quiver(swarm_states[:,1],swarm_states[:,2],swarm_states[:,-2],swarm_states[:,-1],width=0.005, animated=True)
            # Plot single drone + neighbors
            # Apply needed rotations with respect to body frame
            t_x, t_y = swarm_states[selected_drone,:2]
            points = np.array([[-2*self.viewing_radius, 2*self.viewing_radius, 0, 0], 
                                [0, 0, 2*-self.viewing_radius, 2*self.viewing_radius]])
            R_2D = np.array([[swarm_states[selected_drone,-3], -swarm_states[selected_drone,-2]],
                            [swarm_states[selected_drone,-2], swarm_states[selected_drone,-3]]])
            r_points = R_2D @ points
            self.artists["h_dashed"].set_data(r_points[0,:2], r_points[1,:2])
            self.artists["v_dashed"].set_data(r_points[0,2:4], r_points[1,2:4])
            colors = ['#FF0000A0']*(len(neighbors) + 1)
            colors[-1] = '#80ff00'
            if len(neighbors) > 0:
                n_data = np.vstack([n.get_state()[0,0:2] for n in neighbors])
            else:
                n_data = np.array([0,0])

            self.artists["scatter_single"].set(offsets=np.vstack((n_data, [0,0])), color=colors, sizes=sizes[3]*np.ones(len(colors)))
            # Plot heading of estimated neighbors
            if len(neighbors) > 0:
                n_data2 = np.vstack([n.get_state()[3,0:2] for n in neighbors])
                self.artists["quiver_n"] = self.ax[3].quiver(n_data[:,0],n_data[:,1],n_data2[:,0],n_data2[:,1],width=0.005, animated=True)
            # Plot heading of the selected drone
            scale = self.viewing_radius / 5.0
            self.artists["arrow_single"] = self.ax[3].arrow(0, 0, swarm_states[selected_drone,-3], swarm_states[selected_drone,-2], width=scale*0.075, head_width=scale*0.3, head_length=scale*0.25, fc='g', ec='g')
            # Plot estimated and true viewing direction
            self.artists["arrow_est"] = self.ax[3].arrow(0, 0, self._swarm_ref.members[selected_drone].estimated_viewing_dir[0], 
                                                         self._swarm_ref.members[selected_drone].estimated_viewing_dir[1], 
                                                         width=scale*0.075, head_width=scale*0.3, head_length=scale*0.25, fc='#00FF00A0', ec='#00FF00A0')
            self.artists["arrow_true"] = self.ax[3].arrow(0, 0, self._swarm_ref.members[selected_drone].ground_truth_viewing_dir[0], 
                                                          self._swarm_ref.members[selected_drone].ground_truth_viewing_dir[1], 
                                                          width=scale*0.025, head_width=scale*0.3, head_length=scale*0.25, ec='k', fc='#000000A0', linestyle=':', linewidth=1.5)
            # Plot swarm
            self.artists["scatter_swarm"].set(offsets=[swarm_states[i,:2] - np.array([t_x,t_y]) for i in range(self._swarm_ref.count) if i != selected_drone], sizes=sizes[3]*np.ones(self._swarm_ref.count-1))
            # VLOS DEBUG
            if DEBUG_VLOS:
                index = self._swarm_ref.selected_drone
                for i in range(self._swarm_ref.count):
                    self.artists[f'vlos_lines_{i}'], = self.ax[3].plot([0, swarm_states[i,0] - swarm_states[index,0]], [0, swarm_states[i,1] - swarm_states[index,1]], 'k--', alpha=0.5, animated=True)
        except Exception as e:
            print(e)
        return self.artists.values()

    def stop(self):
        """
        Stop the animation
        """
        self.fig.clear()

    def onpick(self, event: tk.Event):
        """
        Callback function for the pick event when selecting a point

        Args:
            event (tk.Event): data of the event
        """
        index = event.ind
        if index is None:
            return

        if (event.artist == self.artists["scatter_swarm"]):
            if index[0] >= self._swarm_ref.selected_drone:
                index[0] += 1
        print("Selected drone: {0}".format(index[0]))
        self._swarm_ref.selected_drone = index[0]
        try:
            self.master.drone_selection_changed()
        except:
            pass

#================================================================================================
# FOR TESTING
#=================================================================================================    
def main():
    root = tk.Tk()
    root.geometry("1000x800")
    root.title("2D Swarm Views")
    swarm = Swarm(10, [0,0,10,5,5,0])
    swarm.set_noise("Uniform",0.2,0.1)
    swarm.members[0].angles = np.array([0,0,np.pi/4])
    swarm.compute_neighborhood("Visual LoS", {"sensing_range":100, "r_agent":0.3})
    r = Renderer2D(root, swarm)
    root.mainloop()

if __name__ == "__main__":
    main()