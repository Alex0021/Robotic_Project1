####################################################
# THIS FILE CONTAINS THE GUI OF THE SIMULATOR
# FOR MY SEMESTER PROJECT
####################################################

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
import sys
from swarm import *
from renderer import *
from simulator import Simulator

w,h = (1600,800)
DEFAULT_NB_DRONES = 10


class myApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.mainframe = root
        self.mainframe.protocol("WM_DELETE_WINDOW", self.app_closing)
        self.mainframe.grid_columnconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(1, weight=3)
        self.mainframe.grid_rowconfigure(0,weight=1)
        self.init_main_panels()
        self.init_sidebar_components()
        # Start 3D plot renderer
        self.renderer = Renderer(self.panel_view, None)

    def init_main_panels(self):
        self.panel_view = tk.Frame(self.mainframe)
        #self.panel_view.grid_columnconfigure(0,weight=1)
        #self.panel_view.grid_rowconfigure(0,weight=1)
        self.panel_view.grid(column=1,row=0,sticky='NWES')

        self.panel_sidebar = tk.Frame(self.mainframe, bg='lightgray')
        self.panel_sidebar.grid_columnconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure(1,weight=6)
        self.panel_sidebar.grid_rowconfigure(2,weight=4)
        self.panel_sidebar.grid(column=0, row=0, sticky='NWES')

        # Subpabels of sidebar
        self.panel_title = tk.Frame(self.panel_sidebar, bg='darkslategray')
        self.panel_title.grid_columnconfigure(0,weight=1)
        self.panel_title.grid_rowconfigure(0,weight=1)
        self.panel_title.grid(column=0,row=0, sticky='NWES')

        self.panel_params = tk.Frame(self.panel_sidebar, borderwidth=2, relief='ridge', padx=5, pady=5)
        self.panel_params.grid_columnconfigure(0, weight=2)
        self.panel_params.grid_columnconfigure((1,2), weight=1)
        self.panel_params.grid_rowconfigure((1,2,3,4),weight=1)
        self.panel_params.grid(column=0, row=1,sticky='NWES')

        self.panel_sim = tk.Frame(self.panel_sidebar, borderwidth=2, relief='ridge', padx=5, pady=5)
        self.panel_sim.grid_columnconfigure(0, weight=2)
        self.panel_sim.grid_columnconfigure((1,2), weight=1)
        self.panel_sim.grid_rowconfigure((0,1,2,3),weight=1)
        self.panel_sim.grid(column=0, row=2,sticky='NWES')

    def init_sidebar_components(self):
        # Title
        self.label_title = ttk.Label(self.panel_title, anchor='center', text="DRONE SWARM BOUNDARIES \n&\n POSE ESTIMATION", justify='center',
                                     font=font.Font(name='Helvetica', weight='bold', size=16), foreground='white', background=self.panel_title['bg'])
        self.label_title.grid(column=0,row=0, sticky='NEWS')

        # Drone number
        self.label_drone_nb = ttk.Label(self.panel_params, anchor='w', text="# drones: ")
        self.label_drone_nb.grid(column=0,row=0, sticky='NEWS')
        self.spinner_drone_nb = ttk.Spinbox(self.panel_params, increment=1,from_=0, to=50)
        self.spinner_drone_nb.set(DEFAULT_NB_DRONES)
        self.spinner_drone_nb.grid(row=0,column=1,sticky='w', padx=5)

        # Swarm spread
        self.label_swarm_spread = ttk.Label(self.panel_params, anchor='w', text="Swarm spread (r): ")
        self.label_swarm_spread.grid(column=0,row=1, sticky='NEWS')
        self.slider_spread = ttk.Scale(self.panel_params, from_=0,to=20, value=10, orient='horizontal')
        self.slider_spread.grid(row=1, column=1, sticky='WE', padx=5)

        # Noise

        # Simulate buttons
        self.btn_init = ttk.Button(self.panel_sim, text="Initialize", command=self._initialize_simulation)
        self.btn_init.grid(column=0,row=0,sticky='WE', padx=10)
        self.btn_simulate = ttk.Button(self.panel_sim, text="Simulate", command=self._btn_simulate_callback)
        self.btn_simulate.grid(column=1,row=0,sticky='WE', padx=10)
        self.btn_pause = ttk.Button(self.panel_sim, text="Pause", command=self._btn_pause_callback)
        self.btn_pause.grid(column=2, row=0, sticky='EW', padx=10)
        self.btn_center = ttk.Button(self.panel_sim, text="Center plot data", command=self._btn_center_callback)
        self.btn_center.grid(column=2, row=1, sticky='EW', padx=10, pady=5)
        self.btn_reset = ttk.Button(self.panel_sim, text="Reset", command=self._button_reset_callback)
        self.btn_reset.grid(column=0, row=1, sticky='EW', padx=10, pady=5)

        # Simulation timestep
        self.panel_sim_timestep = tk.Frame(self.panel_sim)
        self.panel_sim_timestep.grid(column=1, row=1,sticky='NWES')
        self.panel_sim_timestep.grid_columnconfigure(0, weight=1)
        self.panel_sim_timestep.grid_columnconfigure(1, weight=5)
        self.panel_sim_timestep.grid_rowconfigure(0, weight=1)
        self.label_sim_dt = ttk.Label(self.panel_sim_timestep, text="dt: ", justify='right')
        self.label_sim_dt.grid(column=0, row=0, sticky='E')
        self.spinner_sim_dt = ttk.Spinbox(self.panel_sim_timestep, increment=0.001, from_=0.001, to=1)
        self.spinner_sim_dt.set(0.01)
        self.spinner_sim_dt.grid(column=1, row=0, sticky='W', padx=5)

    #==================================#
    #       BUTTON CALLBACKS           #
    #==================================# 
        
    def _initialize_simulation(self):
        # Retrieve all necessary parameters from app widgets
        nb_drones = int(self.spinner_drone_nb.get())
        
        self.swarm = Swarm(count=nb_drones, box=[0,0,10,5,5,5])
        #self.swarm.initialize_random_vel([0.1, 0.5, -0.3, 0.3, 0, 0.2])
        self.swarm.print_swarm()
        #self.swarm.migration_point = np.array([5,0,10])
        dt = float(self.spinner_sim_dt.get())
        self.sim = Simulator(dt, self.swarm)
        self.renderer._swarm_ref = self.swarm
        self.renderer.start()

    def _button_reset_callback(self):
        self.renderer.reset()
        self.renderer._swarm_ref = None
        try:
            self.sim.stop()
        except:
            pass
 
    def _btn_simulate_callback(self):
        self.sim.start()

    def _btn_pause_callback(self):
        self.sim.pause()

    def _btn_center_callback(self):
        self.renderer.center_plot_data()

    def btn_step_callback(self):
        self.sim.step()

    def app_closing(self):
        try:
            self.renderer.stop()
            self.sim.stop()
        except:
            pass
        finally:
            exit()

if __name__ == "__main__":
    root = tk.Tk()
    #if len(sys.argv) > 1:
    #    # Set width and height
    #    root.geometry('{0}x{1}+0+0'.format(sys.argv[1], sys.argv[2]))
    root.title("Swarm boundaries simulation")
    root.geometry('{0}x{1}+0+0'.format(w, h))
    app = myApp(root)

    # TEST CODE #
    #swarm.members[0].vel = np.array([1.0,0,0])
    #swarm.members[0].angles = np.array([0,np.pi/4,-np.pi/4])
    # END TEST #

    root.mainloop()