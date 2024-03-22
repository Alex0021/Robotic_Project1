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
        self.var_swarm_spread = tk.DoubleVar(value=config.get('swarm_spread', 1.0))
        self.var_noise_pos = tk.DoubleVar(value=config.get('noise_pos', 0.025))
        self.var_noise_orient = tk.DoubleVar(value=config.get('noise_orient', 0.025))
        self.var_noise_heading = tk.DoubleVar(value=config.get('noise_heading', 0.0))
        self.var_delta = tk.DoubleVar(value=config.get('delta', 0.0))
        self.var_coh = tk.DoubleVar(value=config.get('r_coh', 0.0))
        self.var_vref = tk.DoubleVar(value=config.get('vref', 0.0))
        self.var_a = tk.DoubleVar(value=config.get('a', 0.0))
        self.var_b = tk.DoubleVar(value=config.get('b', 0.0))
        self.var_c = tk.DoubleVar(value=config.get('c', 0.0))
        self.var_z_offset = tk.DoubleVar(value=config.get('z_offset', 10.0))
        self.var_target = tk.StringVar(value='')
        self.var_neighbor_sampling = tk.IntVar(value=config.get('neighbor_sampling', 1))
        self.spawn_box = config.get('spawn_box', [0,0,10,5,5,5])
        self.cmd_yaw = config.get('cmd_yaw', 0.5)
        self.cmd_vel = config.get('cmd_vel', 0.5)

        #==================================#
        # APP VARIABLES TRACKING  #
        #==================================#
        self.var_neighbor_count.trace_add('write', self._set_neighbors_algo_params)
        self.var_swarm_spread.trace_add('write', self._set_swarm_algo_params)
        self.var_delta.trace_add('write', self._set_swarm_algo_params)
        self.var_coh.trace_add('write', self._set_swarm_algo_params)
        self.var_vref.trace_add('write', self._set_swarm_algo_params)
        self.var_a.trace_add('write', self._set_swarm_algo_params)
        self.var_b.trace_add('write', self._set_swarm_algo_params)
        self.var_c.trace_add('write', self._set_swarm_algo_params)
        self.var_noise_pos.trace_add('write', self.noise_changed_callback)
        self.var_noise_orient.trace_add('write', self.noise_changed_callback)
        self.var_noise_heading.trace_add('write', self.noise_changed_callback)
        self.var_neighbor_sampling.trace_add('write', self._set_neighbors_algo_params)
        #==================================#
        #       APP INITIALIZATION         #
        #==================================#
        self.init_main_panels()
        self.init_sidebar_components()
        self.noise_changed_callback()
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
        self.panel_neighbors = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
        self.panel_neighbors.grid(column=0, row=1, columnspan=3, rowspan=1, sticky='NWES', padx=0, pady=5)
        self.panel_neighbors.grid_columnconfigure((0,1,2,3,4), weight=1)
        self.panel_neighbors.grid_rowconfigure(0, weight=1)
        self.label_neighbors = ttk.Label(self.panel_neighbors, anchor='w', text="Neighbors: ")
        self.label_neighbors.grid(column=0,row=0, sticky='NEWS')
        self.listbox_neighbors_algo = ttk.Combobox(self.panel_neighbors, values=["Eucledian", "Topological", "Voronoi", "Visual LoS"])
        self.listbox_neighbors_algo.set("Topological")
        self.listbox_neighbors_algo.grid(row=0,column=1,sticky='we', padx=5)
        self.listbox_neighbors_algo.bind("<<ComboboxSelected>>", lambda e: self._set_neighbors_algo_params())
        self.label_neighbors_algo_param = ttk.Label(self.panel_neighbors, anchor='e', text="# ")
        self.label_neighbors_algo_param.grid(column=2,row=0,sticky='NEWS')
        self.spinner_neighbors = ttk.Spinbox(self.panel_neighbors, increment=1,from_=0, to=self.var_drone_count.get(), textvariable=self.var_neighbor_count)
        self.spinner_neighbors.grid(row=0,column=3,sticky='w', padx=5)
        self.btn_apply_all_neighbors = ttk.Button(self.panel_neighbors, text="Apply to all", command=lambda e: self._set_neighbors_algo_params(apply_to_all=True))
        self.btn_apply_all_neighbors.grid(row=0,column=4,sticky='wens', padx=5, pady=10)

        # Swarm spread
        self.label_swarm_spread = ttk.Label(self.panel_params, anchor='w', text="Swarm spread (r): ")
        self.label_swarm_spread.grid(column=0,row=2, sticky='NEWS')
        self.slider_spread = ttk.Scale(self.panel_params, from_=0,to=5, orient='horizontal', variable=self.var_swarm_spread, command=lambda val: self.var_swarm_spread.set(round(float(val),2)))
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

        self.label_control_r_coh = ttk.Label(self.panel_control_scheme, anchor='w', text="r_coh: ")
        self.label_control_r_coh.grid(column=2,row=1, sticky='NEWS')
        self.textbox_control_r_coh = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_coh)
        self.textbox_control_r_coh.grid(column=3, row=1, sticky='W', padx=5)

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
        self.pnael_noise = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
        self.pnael_noise.grid(column=0, row=6, columnspan=3, rowspan=2, sticky='NWES', padx=0, pady=5)
        self.pnael_noise.grid_columnconfigure((0,1,2,3,4,5,6), weight=1)
        self.pnael_noise.grid_rowconfigure((0,1), weight=1)
        self.label_noise = ttk.Label(self.pnael_noise, anchor='w', text="Noise: ")
        self.label_noise.grid(column=0,row=0, sticky='NEWS')
        self.listbox_noise_type = ttk.Combobox(self.pnael_noise, values=["None", "Uniform", "Gaussian"])
        self.listbox_noise_type.set("None")
        self.listbox_noise_type.grid(column=1,row=0, sticky='we', padx=5)
        self.listbox_noise_type.bind("<<ComboboxSelected>>", lambda e: self.noise_changed_callback())
        self.label_noise_pos = ttk.Label(self.pnael_noise, anchor='e', text="Dist:", justify='right')
        self.label_noise_pos.grid(column=0,row=1, sticky='NEWS')
        self.spinner_noise_pos = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_pos)
        self.spinner_noise_pos.grid(column=1, row=1, sticky='W', padx=5)
        self.label_noise_orient = ttk.Label(self.pnael_noise, anchor='e', text="Dir: ", justify='right')
        self.label_noise_orient.grid(column=2,row=1, sticky='NEWS')
        self.spinner_noise_orient = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_orient)
        self.spinner_noise_orient.grid(column=3, row=1, sticky='W', padx=5)
        self.label_noise_heading = ttk.Label(self.pnael_noise, anchor='w', text="heading: ", justify='right')
        self.label_noise_heading.grid(column=4,row=1, sticky='NEWS')
        self.spinner_noise_heading = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_heading)
        self.spinner_noise_heading.grid(column=5, row=1, sticky='W', padx=5)
        
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
        self.btn_simulate = ttk.Button(self.panel_sim, text="Simulate", command=self._btn_simulate_callback, state='disabled')
        self.btn_simulate.grid(column=1,row=0,sticky='WE', padx=10)
        self.btn_pause = ttk.Button(self.panel_sim, text="Pause", command=self._btn_pause_callback, state='disabled')
        self.btn_pause.grid(column=2, row=0, sticky='EW', padx=10)
        self.btn_reset = ttk.Button(self.panel_sim, text="Reset", command=self._button_reset_callback)
        self.btn_reset.grid(column=0, row=1, sticky='EW', padx=10, pady=5)
        self.btn_2D_view = ttk.Button(self.panel_sim, text="2D view", command=self.btn_2D_view_callback, state='disabled')
        self.btn_2D_view.grid(column=1, row=1, sticky='EW', padx=10, pady=5)
        self.btn_center = ttk.Button(self.panel_sim, text="Center plot data", command=self._btn_center_callback, state='disabled')
        self.btn_center.grid(column=2, row=1, sticky='EW', padx=10, pady=5)

        # Simulation timestep
        self.panel_sim_timestep = tk.Frame(self.panel_sim)
        self.panel_sim_timestep.grid(column=0, row=2,sticky='NWES')
        self.panel_sim_timestep.grid_columnconfigure(0, weight=1)
        self.panel_sim_timestep.grid_columnconfigure(1, weight=5)
        self.panel_sim_timestep.grid_rowconfigure(0, weight=1)
        self.label_sim_dt = ttk.Label(self.panel_sim_timestep, text="dt: ", justify='right')
        self.label_sim_dt.grid(column=0, row=0, sticky='E')
        self.spinner_sim_dt = ttk.Spinbox(self.panel_sim_timestep, increment=0.001, from_=0.001, to=1)
        self.spinner_sim_dt.set(0.01)
        self.spinner_sim_dt.grid(column=1, row=0, sticky='W', padx=5)

        # Neihbors display
        self.panel_sim_neighbors = tk.Frame(self.panel_sim)
        self.panel_sim_neighbors.grid_columnconfigure((0,1), weight=1)
        self.panel_sim_neighbors.grid_columnconfigure(2, weight=2)
        self.panel_sim_neighbors.grid_rowconfigure(0, weight=1)
        self.panel_sim_neighbors.grid(row=2, column=1, columnspan=2, sticky='NWES')
        self.label_neighbor_sampling = ttk.Label(self.panel_sim_neighbors, text="Neighbor sampling: ", justify='right')
        self.label_neighbor_sampling.grid(column=0, row=0, sticky='E')
        self.spinner_neighbor_sampling = ttk.Spinbox(self.panel_sim_neighbors, increment=1, from_=1, to=100, textvariable=self.var_neighbor_sampling)
        self.spinner_neighbor_sampling.grid(column=1, row=0, sticky='W', padx=5)
        self.listbox_neighbors_select = ttk.Combobox(self.panel_sim_neighbors, values=["None", "Selected", "All"])
        self.listbox_neighbors_select.set("None")
        self.listbox_neighbors_select.bind("<<ComboboxSelected>>", lambda e: self._set_neighbors_algo_params())
        self.listbox_neighbors_select.grid(column=2, row=0, sticky='WE', padx=5)


    #==================================#
    #       BUTTON CALLBACKS           #
    #==================================# 
        
    def _initialize_simulation(self):
        # Retrieve all necessary parameters from app widgets
        nb_drones = self.var_drone_count.get()
        swarm_spread = self.slider_spread.get()
        noise = {'type': self.listbox_noise_type.get(), 'param_pos': self.var_noise_pos.get(), 'param_heading': self.var_noise_orient.get()}
        if self.var_target.get() == '':
            target = None
        else:
            target_numbers = self.var_target.get().split(';')
            target = np.asarray(target_numbers, dtype=float)

        self.swarm = Swarm(count=nb_drones, box=self.spawn_box, noise=noise, migration_point=target)
        #self.swarm.initialize_random_vel([0.1, 0.5, -0.3, 0.3, 0, 0.2])
        self._set_neighbors_algo_params()
        self._set_swarm_algo_params()
        self.swarm.print_swarm()
        print("Initializing swarm with parameters: ")
        print(self.swarm.algo_params)
        #self.swarm.migration_point = np.array([5,0,10])
        dt = float(self.spinner_sim_dt.get())
        self.sim = Simulator(dt, self.swarm)
        self.renderer._swarm_ref = self.swarm
        self.renderer.start()
        #self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,0.2]))
        # Unlock buttons
        self.btn_2D_view.config(state='normal')
        self.btn_simulate.config(state='normal')
        self.btn_center.config(state='normal')

    def _set_swarm_algo_params(self, *args):
        if self.swarm is None:
            return
        try:
            self.swarm.algo_params.update( {
                'delta': self.var_delta.get(),
                'd_ref': self.var_swarm_spread.get(),
                'a': self.var_a.get(),
                'b': self.var_b.get(),
                'r0_coh': self.var_coh.get()
            })
        except ValueError as e:
            pass
        except Exception as e:
            print("Error setting swarm algo params: {0}".format(e))

    def _set_neighbors_algo_params(self, apply_to_all=False, *args):
        if self.swarm is None:
            return
        try:
            self.swarm.update_neighbors_metric({
                'computation': self.listbox_neighbors_select.get(),
                'metric': self.listbox_neighbors_algo.get(),
                'sampling': self.var_neighbor_sampling.get(),
                'metric_specific': {'nb_neighbors': self.var_neighbor_count.get()}
            }, apply_to_all)
        except Exception as e:
            print("Error setting neighbors algo params: {0}".format(e))

    def _button_reset_callback(self):
        self.renderer.reset()
        self.renderer._swarm_ref = None
        # Disabling simulation controls
        self.btn_center.config(state='disabled')
        self.btn_simulate.config(state='disabled')
        self.btn_2D_view.config(state='disabled')
        try:
            self.sim.stop()
            self.sim = None
        except:
            pass
 
    def _btn_simulate_callback(self):
        self.btn_pause.config(state='normal')
        self.btn_simulate.config(state='disabled')
        self.sim.start()

    def _btn_pause_callback(self):
        self.btn_simulate.config(state='normal')
        self.btn_pause.config(state='disabled')
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


    def noise_changed_callback(self, *args):
        if self.listbox_noise_type.get() == 'None':
            self.spinner_noise_pos.config(state='disabled')
            self.spinner_noise_orient.config(state='disabled')
            self.spinner_noise_heading.config(state='disabled')
        else:
            self.spinner_noise_pos.config(state='normal')
            self.spinner_noise_orient.config(state='normal')
            self.spinner_noise_heading.config(state='normal')
        if self.swarm is None:
            return
        try:
            self.swarm.set_noise(self.listbox_noise_type.get(), self.var_noise_pos.get(), self.var_noise_orient.get())
            print('Noise parameters changed: {0}'.format(self.swarm.noise))
        except Exception as e:
            print("Error setting noise: {0}".format(e))

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