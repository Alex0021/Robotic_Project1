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
import json

w,h = (1600,800)
DEFAULT_NB_DRONES = 10
CONFIG_FILENAME = 'app_config.json'


class myApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.mainframe = root

        #==================================#
        #        EVENT HANDLERS            #
        #==================================#
        self.mainframe.bind("<KeyPress>", self.key_press_callback)
        self.mainframe.bind("<KeyRelease>", self.key_release_callback)
        self.mainframe.protocol("WM_DELETE_WINDOW", self.app_closing)
        self.mainframe.grid_columnconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(1, weight=3)
        self.mainframe.grid_rowconfigure(0,weight=1)

        #==================================#
        #       APP VARIABLES              #
        #==================================#
        self.var_drone_count = tk.IntVar(value=DEFAULT_NB_DRONES)
        self.var_neighbor_count = tk.IntVar(value= DEFAULT_NB_DRONES-1)
        self.var_swarm_spread = tk.DoubleVar(value=10.0)
        self.var_noise_pos = tk.DoubleVar()
        self.var_noise_orient = tk.DoubleVar()
        self.var_delta = tk.DoubleVar()
        self.var_dref = tk.DoubleVar()
        self.var_vref = tk.DoubleVar()
        self.var_a = tk.DoubleVar()
        self.var_b = tk.DoubleVar()
        self.var_c = tk.DoubleVar()
        self.var_z_offset = tk.DoubleVar(value=10.0)
        self.var_target = tk.StringVar(value='{0};{1};{2}'.format(5.0,5.0,round(self.var_z_offset.get(),1)))

        self.window_2d = None
        self.swarm = None

        #==================================#
        # JSON CONFIG FILE INITIALIZATION  #
        #==================================#
        try:
            with open(CONFIG_FILENAME) as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        # Initialize app variables with json values or defaults
        self.var_drone_count = tk.IntVar(value=config.get('drone_count', DEFAULT_NB_DRONES))
        self.var_neighbor_count = tk.IntVar(value=config.get('neighbor_count', DEFAULT_NB_DRONES-1))
        self.var_swarm_spread = tk.DoubleVar(value=config.get('swarm_spread', 10.0))
        self.var_noise_pos = tk.DoubleVar(value=config.get('noise_pos', 0.0))
        self.var_noise_orient = tk.DoubleVar(value=config.get('noise_orient', 0.0))
        self.var_delta = tk.DoubleVar(value=config.get('delta', 0.0))
        self.var_dref = tk.DoubleVar(value=config.get('dref', 0.0))
        self.var_vref = tk.DoubleVar(value=config.get('vref', 0.0))
        self.var_a = tk.DoubleVar(value=config.get('a', 0.0))
        self.var_b = tk.DoubleVar(value=config.get('b', 0.0))
        self.var_c = tk.DoubleVar(value=config.get('c', 0.0))
        self.var_z_offset = tk.DoubleVar(value=config.get('z_offset', 10.0))
        self.var_target = tk.StringVar(value='')
        self.spawn_box = config.get('spawn_box', [0,0,10,5,5,5])
        self.cmd_yaw = config.get('cmd_yaw', 0.5)
        self.cmd_vel = config.get('cmd_vel', 0.5)
        #==================================#
        #       APP INITIALIZATION         #
        #==================================#
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
        self.panel_params.grid_rowconfigure(list(range(9)),weight=1)
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
        self.spinner_drone_nb = ttk.Spinbox(self.panel_params, increment=1,from_=1, to=50, textvariable=self.var_drone_count, command=self._update_neighbors_spinbox)
        self.spinner_drone_nb.grid(row=0,column=1,sticky='w', padx=5)

        # Neighbors
        self.label_neighbors = ttk.Label(self.panel_params, anchor='w', text="# Neighbors: ")
        self.label_neighbors.grid(column=0,row=1, sticky='NEWS')
        self.spinner_neighbors = ttk.Spinbox(self.panel_params, increment=1,from_=0, to=self.var_drone_count.get(), textvariable=self.var_neighbor_count)
        self.spinner_neighbors.grid(row=1,column=1,sticky='w', padx=5)
        self.listbox_neighbors_algo = ttk.Combobox(self.panel_params, values=["Eucledian", "Topological", "Voronoi", "Visual LoS"])
        self.listbox_neighbors_algo.set("Topological")
        self.listbox_neighbors_algo.grid(row=1,column=2,sticky='we', padx=5)

        # Swarm spread
        self.label_swarm_spread = ttk.Label(self.panel_params, anchor='w', text="Swarm spread (r): ")
        self.label_swarm_spread.grid(column=0,row=2, sticky='NEWS')
        self.slider_spread = ttk.Scale(self.panel_params, from_=0,to=30, orient='horizontal', variable=self.var_swarm_spread, command=lambda val: self.var_swarm_spread.set(round(float(val),2)))
        self.slider_spread.grid(row=2, column=1, sticky='WE', padx=5)
        self.textbox_slider_value = ttk.Entry(self.panel_params, textvariable=self.var_swarm_spread, width=10)
        self.textbox_slider_value.grid(row=2, column=2, sticky='W', padx=5)

        # Control params panel
        self.panel_control_scheme = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
        self.panel_control_scheme.grid(column=0, row=3, columnspan=3, rowspan=3, sticky='NWES', padx=0, pady=5)
        self.panel_control_scheme.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        self.panel_control_scheme.grid_rowconfigure((0,1,2), weight=1)
        # COntrol scheme
        self.label_control_scheme = ttk.Label(self.panel_control_scheme, anchor='w', text="Control scheme: ")
        self.label_control_scheme.grid(column=0,row=0, columnspan=2, sticky='NEWS')
        self.listbox_control_scheme = ttk.Combobox(self.panel_control_scheme, values=["Olfati-Saber"])
        self.listbox_control_scheme.set("Olfati-Saber")
        self.listbox_control_scheme.grid(row=0,column=2,columnspan=2, sticky='we', padx=5)
        # Control variables
        self.label_control_delta = ttk.Label(self.panel_control_scheme, anchor='w', text="delta: ")
        self.label_control_delta.grid(column=0,row=1, sticky='NEWS')
        self.textbox_control_delta = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_delta)
        self.textbox_control_delta.grid(column=1, row=1, sticky='W', padx=5)

        self.label_control_dref = ttk.Label(self.panel_control_scheme, anchor='w', text="dref: ")
        self.label_control_dref.grid(column=2,row=1, sticky='NEWS')
        self.textbox_control_dref = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_dref)
        self.textbox_control_dref.grid(column=3, row=1, sticky='W', padx=5)

        self.label_control_vref = ttk.Label(self.panel_control_scheme, anchor='w', text="vref: ")
        self.label_control_vref.grid(column=4,row=1, sticky='NEWS')
        self.textbox_control_vref = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_vref)
        #self.textbox_control_vref.bind("<Return>", lambda e: self.swarm.set_cmd_velocity(np.array([float(self.var_vref.get())]*3)))
        self.textbox_control_vref.grid(column=5, row=1, sticky='W', padx=5)

        self.label_control_a = ttk.Label(self.panel_control_scheme, anchor='w', text="a: ")
        self.label_control_a.grid(column=0,row=2, sticky='NEWS')
        self.textbox_control_a = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_a)
        self.textbox_control_a.grid(column=1, row=2, sticky='W', padx=5)

        self.label_control_b = ttk.Label(self.panel_control_scheme, anchor='w', text="b: ")
        self.label_control_b.grid(column=2,row=2, sticky='NEWS')
        self.textbox_control_b = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_b)
        self.textbox_control_b.grid(column=3, row=2, sticky='W', padx=5)

        self.label_control_c = ttk.Label(self.panel_control_scheme, anchor='w', text="c: ")
        self.label_control_c.grid(column=4,row=2, sticky='NEWS')
        self.textbox_control_c = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_c)
        self.textbox_control_c.grid(column=5, row=2, sticky='W', padx=5)
        
        # Noise
        self.pnael_noise = tk.Frame(self.panel_params)
        self.pnael_noise.grid(column=0, row=6, columnspan=3, sticky='NWES', padx=0, pady=5)
        self.pnael_noise.grid_columnconfigure((0,1,2,3,4,5,6), weight=1)
        self.pnael_noise.grid_rowconfigure(0, weight=1)
        self.label_noise = ttk.Label(self.pnael_noise, anchor='w', text="Noise: ")
        self.label_noise.grid(column=0,row=0, sticky='NEWS')
        self.listbox_noise_type = ttk.Combobox(self.pnael_noise, values=["None", "Uniform", "Gaussian"])
        self.listbox_noise_type.set("None")
        self.listbox_noise_type.grid(column=1,row=0, sticky='we', padx=5)
        self.label_noise_pos = ttk.Label(self.pnael_noise, anchor='e', text="Position :", justify='right')
        self.label_noise_pos.grid(column=2,row=0, sticky='NEWS')
        self.textbox_noise_pos = ttk.Entry(self.pnael_noise, width=10, textvariable=self.var_noise_pos)
        self.textbox_noise_pos.grid(column=3, row=0, sticky='W', padx=5)
        self.label_noise_orient = ttk.Label(self.pnael_noise, anchor='e', text="Orientation: ", justify='right')
        self.label_noise_orient.grid(column=4,row=0, sticky='NEWS')
        self.textbox_noise_orient = ttk.Entry(self.pnael_noise, width=10, textvariable=self.var_noise_orient)
        self.textbox_noise_orient.grid(column=5, row=0, sticky='W', padx=5)

        # Z-offset
        self.label_z_offset = ttk.Label(self.panel_params, anchor='w', text="Z-offset: ")
        self.label_z_offset.grid(column=0,row=7, sticky='NEWS')
        self.spinner_z_offset = ttk.Spinbox(self.panel_params, increment=0.1,from_=0, to=10, textvariable=self.var_z_offset)
        self.spinner_z_offset.grid(row=7,column=1,sticky='w', padx=5)
        
        # Target
        self.label_target = ttk.Label(self.panel_params, anchor='w', text="Target: ")
        self.label_target.grid(column=0,row=8, sticky='NEWS')
        self.textbox_target = ttk.Entry(self.panel_params, textvariable=self.var_target, font=font.Font(size=10))
        self.textbox_target.grid(column=1, row=8, sticky='WE', padx=5)
        self.label_target_format = ttk.Label(self.panel_params, anchor='w', text="(#.#;#.#;#.#) or empty", justify='left', font=font.Font(size=10))
        self.label_target_format.grid(column=2,row=8, sticky='NEWS')

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
        self.btn_2D_view = ttk.Button(self.panel_sim, text="2D view", command=self.btn_2D_view_callback, state='disabled')
        self.btn_2D_view.grid(column=2, row=2, sticky='EW', padx=10, pady=5)
        self.btn_show_neighbors = ttk.Button(self.panel_sim, text="Toggle neighbors", command=self.btn_show_neighbors_callback)
        self.btn_show_neighbors.grid(column=1, row=2, sticky='EW', padx=10, pady=5)


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
        nb_drones = self.var_drone_count.get()
        neighbors = self.var_neighbor_count.get()
        neighbors_algo = self.listbox_neighbors_algo.get()
        swarm_spread = self.slider_spread.get()
        noise = {'type': self.listbox_noise_type.get(), 'param_pos': self.var_noise_pos.get(), 'param_heading': self.var_noise_orient.get()}
        if self.var_target.get() == '':
            target = None
        else:
            target_numbers = self.var_target.get().split(';')
            target = np.asarray(target_numbers, dtype=float)
        
        algo_params = {
            'delta': self.var_delta.get(),
            'd_ref': self.var_dref.get(),
            'v_ref': np.zeros(3),
            'a': self.var_a.get(),
            'b': self.var_b.get(),
            'r0_coh': swarm_spread,
            'neighborhood_metric': neighbors_algo,
            'neighborhood_metric_data': {'nb_neighbors': neighbors}
        }
        self.swarm = Swarm(count=nb_drones, box=self.spawn_box, algo_params=algo_params, noise=noise, migration_point=target)
        #self.swarm.initialize_random_vel([0.1, 0.5, -0.3, 0.3, 0, 0.2])
        self.swarm.print_swarm()
        print("Initializing swarm with parameters: ")
        print(algo_params)
        #self.swarm.migration_point = np.array([5,0,10])
        dt = float(self.spinner_sim_dt.get())
        self.sim = Simulator(dt, self.swarm)
        self.renderer._swarm_ref = self.swarm
        self.renderer.start()
        #self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,0.2]))
        # Unlock buttons
        self.btn_2D_view.config(state='normal')

    def _button_reset_callback(self):
        self.renderer.reset()
        self.renderer._swarm_ref = None
        try:
            self.sim.stop()
            self.sim = None
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

    def btn_2D_view_callback(self):
        # Initialize 2D view windows
        if self.window_2d is None:
            self.window_2d = tk.Toplevel(self.mainframe)
            self.window_2d.protocol("WM_DELETE_WINDOW", self.window_2d_closing)
            self.window_2d.title("2D Swarm viewer")
            self.window_2d.geometry('1000x800')
            self.renderer2D = Renderer2D(self.window_2d, self.swarm)
        self.window_2d.deiconify()

    def btn_show_neighbors_callback(self):
        self.renderer.show_neighbors = not self.renderer.show_neighbors
        self.swarm.is_computing_neighborhood = self.renderer.show_neighbors

    def key_press_callback(self, event):
        if self.swarm is None:
            return
        target_vel = self.swarm.get_cmd_velocity()
        match event.char:
            case 'q':
                self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,self.cmd_yaw]))
            case 'e':
                self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,-self.cmd_yaw]))
            case 'w':
                target_vel[0] = self.var_vref.get()
            case 's':
                target_vel[0] = -self.var_vref.get()
            case 'a':
                target_vel[1] = self.var_vref.get()
            case 'd':
                target_vel[1] = -self.var_vref.get()
        self.swarm.set_cmd_velocity(target_vel)   


    def key_release_callback(self, event):
        if self.swarm is None:
            return
        target_vel = self.swarm.get_cmd_velocity()
        match event.char:
            case 'q':
                self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,0.0]))
            case 'e':
                self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,0.0]))
            case 'w':
                target_vel[0] = 0
            case 's':
                target_vel[0] = 0
            case 'a':
                target_vel[1] = 0
            case 'd':
                target_vel[1] = 0
        self.swarm.set_cmd_velocity(target_vel)    

    def app_closing(self):
        try:
            self.renderer.stop()
            self.sim.stop()
        except:
            pass
        finally:
            exit()

    def window_2d_closing(self):
        self.window_2d.destroy()
        self.window_2d = None
        self.renderer2D.stop()
        self.renderer2D = None

    def _update_neighbors_spinbox(self):
        self.spinner_neighbors.config(to=self.var_drone_count.get()-1)
        if self.var_neighbor_count.get() >= self.var_drone_count.get():
            self.var_neighbor_count.set(self.var_drone_count.get()-1)

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