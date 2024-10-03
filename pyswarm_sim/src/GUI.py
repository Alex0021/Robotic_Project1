####################################################
# THIS FILE CONTAINS THE GUI OF THE SIMULATOR
####################################################
import sys, os
# sys.path.insert(0, './src')

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from pyswarm_sim.src.swarm import *
from pyswarm_sim.src.renderer import *
from pyswarm_sim.src.simulator import Simulator
from pyswarm_sim.src.recorder import SwarmRecorder
import json
import time
from pyswarm_sim.src.autorun import AutorunSim
from pyswarm_sim.src.panels.control_scheme_panel import ControlSchemePanel
from pyswarm_sim.src.environment import Environment

os.path.join(os.path.dirname(__file__), 'pyswarm_sim')

w,h = (1600,800)
CONFIG_FILENAME = 'app_config.json'     # Default config filename
DEFAULT_CONFIG_FILENAME = 'app_config_template.json'
AUTORUN_FILENAME = 'sim_autorun.json'   # Default auorun filename
DATA_OUTPUT_FOLDER = 'sim_results'      # Output folder for .npy data files after recording
FRONTEND_UPDATE_INTERVAL = 0.1          # To querry updates for the algo panel sidebar (viewing directions, coverage %)


class myApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.mainframe = root

        #==================================#
        #        APP CONSTANTS             #
        #==================================#
        self.RENDERER_CALLBACKS = {
            'drone_selection_changed': self.drone_selection_changed,
            'obstacle_moved_callback': self.obstacle_moved_callback,
            'obstacle_clicked_callback': self.obstacle_clicked_callback
        }

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
        self.renderer2D = None
        self.swarm = None
        self.autorun = None

        #==================================#
        # JSON CONFIG FILE INITIALIZATION  #
        #==================================#
        try:
            with open(os.path.join('./pyswarm_sim/config', CONFIG_FILENAME)) as f:
                self.app_config = json.load(f)
        except FileNotFoundError:
            # Default config file
            # Create app config file if not found
            print(f'!! Config file not found. Creating default config file {CONFIG_FILENAME} from {DEFAULT_CONFIG_FILENAME}')
            with open(os.path.join('./pyswarm_sim/config', DEFAULT_CONFIG_FILENAME)) as f:
                self.app_config = json.load(f)
            with open(os.path.join('./pyswarm_sim/config', CONFIG_FILENAME), 'w') as f:
                json.dump(self.app_config, f, indent=4)

        # Initialize app variables with json values or defaults
        self.var_drone_count = tk.IntVar(value=self.app_config.get('drone_count', 10))
        self.var_neighbors_metric = tk.StringVar(value=self.app_config['neighbors'].get('metric', 'Topological'))
        self.var_neighbors_count = tk.IntVar(value=self.app_config['neighbors'].get('count', 1))
        self.var_neighbors_sensing_range = tk.DoubleVar(value=self.app_config['neighbors'].get('sensing_range', 1.0))
        self.var_neighbors_r_agent = tk.DoubleVar(value=self.app_config['neighbors'].get('r_agent', 0.01))
        self.var_neighbors_sampling = tk.IntVar(value=self.app_config['neighbors'].get('sampling', 1))
        self.var_noise_type = tk.StringVar(value=self.app_config['noise'].get('distribution', 'None'))
        self.var_noise_param_dist = tk.DoubleVar(value=self.app_config['noise'].get('param_dist', 0.05))
        self.var_noise_param_dir = tk.DoubleVar(value=self.app_config['noise'].get('param_dir', 0.02))
        self.var_noise_param_heading = tk.DoubleVar(value=self.app_config['noise'].get('param_heading', 0.0))
        self.var_z_offset = tk.DoubleVar(value=self.app_config.get('z_offset', 10.0))
        self.spawn_area = self.app_config.get('spawn_area', [0,0,0,1])
        self.cmd_yaw = self.app_config.get('cmd_yaw', 0.5)
        self.cmd_vel = self.app_config.get('cmd_vel', 0.5)
        self.var_viewing_metric_outer_points = tk.IntVar(value=self.app_config['viewing_metric'].get('outer_points', 2))
        self.var_output_csv = tk.StringVar(value=self.app_config.get('output_data', 'output'))
        self.render_env = self.app_config['simulation'].get('render', True)
        self.var_trajectory_mode = tk.StringVar(value=self.app_config['simulation'].get('trajectory_mode', 'target'))
        self.var_swarming_algorithm = tk.StringVar(value=self.app_config['swarming_algorithm'].get('name', 'olfati-saber'))
        self.var_viewing_metric_dim = tk.IntVar(value=self.app_config['viewing_metric'].get('dim', 2))
        self.var_viewing_metric_algorithm = tk.StringVar(value=self.app_config['viewing_metric'].get('algorithm', 'None'))
        self.var_viewing_metric_faces = tk.StringVar(value=self.app_config['viewing_metric'].get('faces', 'adjacent'))
        self.var_sim_dt = tk.DoubleVar(value=self.app_config['simulation'].get('dt', 0.01))
        self.var_autorun_filename = tk.StringVar(value=AUTORUN_FILENAME)

        #==================================#
        #   APP VARIABLES TRACKING         #
        #==================================#
        self.var_drone_count.trace_add('write', self._update_swarm_count)
        self.var_neighbors_metric.trace_add('write', self._update_neighbors_panel_components)
        self.var_neighbors_count.trace_add('write', self._set_neighbors_algo_params)
        self.var_swarming_algorithm.trace_add('write', self.swarming_algo_changed)
        self.var_noise_type.trace_add('write', self.noise_changed_callback)
        self.var_noise_param_dist.trace_add('write', self.noise_changed_callback)
        self.var_noise_param_dir.trace_add('write', self.noise_changed_callback)
        self.var_noise_param_heading.trace_add('write', self.noise_changed_callback)
        self.var_neighbors_sampling.trace_add('write', self._set_neighbors_algo_params)
        self.var_neighbors_sensing_range.trace_add('write', self._set_neighbors_algo_params)
        self.var_neighbors_r_agent.trace_add('write', self._set_neighbors_algo_params)
        self.var_viewing_metric_outer_points.trace_add('write', self.viewing_metric_changed_callback)
        self.var_viewing_metric_algorithm.trace_add('write', self.viewing_metric_changed_callback)
        self.var_viewing_metric_faces.trace_add('write', self.viewing_metric_changed_callback)
        self.var_sim_dt.trace_add('write', self._update_sim_dt)

        #==================================#
        #       APP INITIALIZATION         #
        #==================================#
        self.init_main_panels()
        self.init_algo_components()
        self.init_sidebar_components()
        self.init_env_components()
        self.init_concave_hull_components()
        self.noise_changed_callback()

        # Initialize environment
        self.env = Environment(self.app_config['obstacles'], self.app_config.get('target', None))
        # Start 3D plot renderer
        if self.render_env:
            self.renderer = Renderer(self.panel_view, None, self.env, callbacks=self.RENDERER_CALLBACKS)
        self._generate_obstacles_list()
        # Check for target
        self.textbox_target.delete(0, tk.END)
        self.textbox_target.insert(0, self.app_config.get('target', ''))
        self.swarm_2d = self.app_config.get('swarm_2d', True)
        self.swarm_traj = self.app_config['simulation'].get('trajectory', 'circle')
        self._is_autorun = False
        # Create recorder object
        self._recorder = SwarmRecorder(self.swarm, float(self.var_sim_dt.get()), verbose=True)


    def init_main_panels(self):
        self.panel_view = tk.Frame(self.mainframe)
        #self.panel_view.grid_columnconfigure(0,weight=1)
        #self.panel_view.grid_rowconfigure(0,weight=1)
        self.panel_view.grid(column=1,row=0,sticky='NWES')
        self.panel_view.grid_columnconfigure(0, weight=1)
        self.panel_view.grid_rowconfigure(0, weight=1)
        self.label_no_renderering = ttk.Label(self.panel_view, text='RENDERING DISABLED', font=font.Font(size=20, weight='bold'), justify='center', anchor='center')
        if not self.render_env:
            self.label_no_renderering.grid(column=0,row=0,sticky='NESW')

        # Creating app tabs
        self.tabbed_pane = ttk.Notebook(self.mainframe)
        self.tabbed_pane.grid(column=0,row=0,sticky='NWES')
        self.tabbed_pane.bind("<<NotebookTabChanged>>", self._tabbed_panel_changed)

        # Panel global params
        self.panel_sidebar = tk.Frame(self.mainframe, bg='lightgray')
        self.panel_sidebar.grid_columnconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure(1,weight=6)
        self.panel_sidebar.grid_rowconfigure(2,weight=3)
        self.tabbed_pane.add(self.panel_sidebar, text='Swarm config')

        # Panel environment
        self.panel_env = tk.Frame(self.mainframe, bg='lightgray')
        self.panel_env.grid_columnconfigure(0,weight=1)
        self.panel_env.grid_rowconfigure(1,weight=1)
        self.tabbed_pane.add(self.panel_env, text='Environment')

        # Panel algo params
        self.panel_algo = tk.Frame(self.mainframe, bg='lightgray')
        self.panel_algo.grid_columnconfigure((0,1,2),weight=1)
        self.panel_algo.grid_rowconfigure(list(range(10)),weight=1)
        self.panel_algo.grid_rowconfigure(10,weight=2)
        self.tabbed_pane.add(self.panel_algo, text='Algo params')

        # Panel concave hull
        self.panel_hull = tk.Frame(self.mainframe)
        self.panel_hull.grid_columnconfigure(0,weight=1)
        self.tabbed_pane.add(self.panel_hull, text='Concave hull')
        # Subpabels of sidebar
        self.panel_title = tk.Frame(self.panel_sidebar, bg='darkslategray')
        self.panel_title.grid_columnconfigure(0,weight=1)
        self.panel_title.grid_rowconfigure(0,weight=1)
        self.panel_title.grid(column=0,row=0, sticky='NWES')

        self.panel_params = tk.Frame(self.panel_sidebar, borderwidth=2, relief='ridge', padx=0, pady=0)
        self.panel_params.grid_columnconfigure(0, weight=2)
        self.panel_params.grid_columnconfigure((1,2), weight=1)
        self.panel_params.grid_rowconfigure((2,3,4,5),weight=1)
        self.panel_params.grid(column=0, row=1,sticky='NWES')

        self.panel_sim = tk.Frame(self.panel_sidebar, borderwidth=2, relief='ridge', padx=5, pady=0)
        self.panel_sim.grid_columnconfigure(0, weight=2)
        self.panel_sim.grid_columnconfigure((1,2), weight=1)
        self.panel_sim.grid_rowconfigure((0,1,3),weight=1)
        self.panel_sim.grid(column=0, row=2,sticky='NWES')

    def init_algo_components(self):
        self.label_algo = ttk.Label(self.panel_algo, anchor='w', text="Algorithm choice: ", justify='left', font=font.Font(size=14))
        self.label_algo.grid(column=0,row=0, sticky='NEWS')
        self.listbox_viewing_algo = ttk.Combobox(self.panel_algo, values=["None", "average", "outer", "tangent_plane", "convex_hull", "alpha_shape"], state='disabled', font=font.Font(size=14))
        self.listbox_viewing_algo.set(self.var_viewing_metric_algorithm.get())
        self.listbox_viewing_algo.grid(column=1,row=0, sticky='W', padx=5)
        self.listbox_viewing_algo.bind("<<ComboboxSelected>>", lambda _: self.var_viewing_metric_algorithm.set(self.listbox_viewing_algo.get()))
        self.label_algo_param = ttk.Label(self.panel_algo, anchor='w', text="Algo params:", justify='left', font=font.Font(size=14))
        self.label_algo_param.grid(column=0,row=1, sticky='NEWS')

        # Algo params for outer metric
        self.panel_algo_params_outer = tk.Frame(self.panel_algo, borderwidth=2, relief='ridge')
        self.panel_algo_params_outer.grid_columnconfigure((0,1), weight=1)
        self.panel_algo_params_outer.grid_rowconfigure(0, weight=1)
        self.label_algo_param_outer_points = ttk.Label(self.panel_algo_params_outer, anchor='w', text="# points: ", font=font.Font(size=14))
        self.label_algo_param_outer_points.grid(column=0,row=0, sticky='NEWS')
        self.spinner_algo_param_outer_points = ttk.Spinbox(self.panel_algo_params_outer, increment=1, from_=2, to=10, textvariable=self.var_viewing_metric_outer_points, font=font.Font(size=12), width=20)
        self.spinner_algo_param_outer_points.grid(column=1,row=0, sticky='W', padx=5)

        # Algo params for convex hull metric
        self.panel_algo_params_convex_hull = tk.Frame(self.panel_algo, borderwidth=2, relief='ridge')
        self.panel_algo_params_convex_hull.grid_columnconfigure((0,1), weight=1)
        self.panel_algo_params_convex_hull.grid_rowconfigure(0, weight=1)
        self.label_algo_param_convex_hull_face = ttk.Label(self.panel_algo_params_convex_hull, anchor='w', text="Normal faces: ", font=font.Font(size=14))
        self.label_algo_param_convex_hull_face.grid(column=0,row=0, sticky='NEWS')
        self.listbox_convex_hull_faces = ttk.Combobox(self.panel_algo_params_convex_hull, values=["adjacent", "visible"], font=font.Font(size=12))
        self.listbox_convex_hull_faces.set(self.var_viewing_metric_faces.get())
        self.listbox_convex_hull_faces.grid(column=1,row=0, sticky='W', padx=5)
        self.listbox_convex_hull_faces.bind("<<ComboboxSelected>>", lambda _: self.var_viewing_metric_faces.set(self.listbox_convex_hull_faces.get()))

        # Choose between 2D/3D viewing direction
        self.label_viewing_dir_dim = ttk.Label(self.panel_algo, anchor='w', text="Viewing dir dim: ", font=font.Font(size=14))
        self.label_viewing_dir_dim.grid(column=0,row=2, sticky='NEWS')
        self.panel_viewing_dir_dim = tk.Frame(self.panel_algo)
        self.panel_viewing_dir_dim.grid_columnconfigure((0,1,2), weight=1)
        self.panel_viewing_dir_dim.grid_rowconfigure(0, weight=1)
        self.panel_viewing_dir_dim.grid(row=2, column=1, sticky='NEWS', padx=5)
        self.radio_2D_viewing = ttk.Radiobutton(self.panel_viewing_dir_dim, text="2D", variable=self.var_viewing_metric_dim, value=2, command=self.viewing_metric_changed_callback, state='disabled')
        self.radio_2D_viewing.grid(column=0,row=0, sticky='NEWS')
        self.radio_3D_viewing = ttk.Radiobutton(self.panel_viewing_dir_dim, text="3D", variable=self.var_viewing_metric_dim, value=3, command=self.viewing_metric_changed_callback, state='disabled')
        self.radio_3D_viewing.grid(column=1,row=0, sticky='NEWS')

        # Show stats for viewing direction
        self.label_estimated_viewing_dir = ttk.Label(self.panel_algo, anchor='w', text="Estimated viewing dir: ", font=font.Font(size=14))
        self.label_estimated_viewing_dir.grid(column=0,row=3, sticky='NEWS')
        self.label_estimated_viewing_dir_value = ttk.Label(self.panel_algo, anchor='w', text="[0.0;0.0;0.0]", font=font.Font(size=16))
        self.label_estimated_viewing_dir_value.grid(column=1,row=3, sticky='NEWS', padx=5)
        self.label_true_viewing_dir = ttk.Label(self.panel_algo, anchor='w', text='"True" viewing dir: ', font=font.Font(size=14))
        self.label_true_viewing_dir.grid(column=0,row=4, sticky='NEWS')
        self.label_true_viewing_dir_value = ttk.Label(self.panel_algo, anchor='w', text="[0.0;0.0;0.0]", font=font.Font(size=16))
        self.label_true_viewing_dir_value.grid(column=1,row=4, sticky='NEWS', padx=5)
        self.label_error_viewing_dir = ttk.Label(self.panel_algo, anchor='w', text="Estimated error: ", font=font.Font(size=14))
        self.label_error_viewing_dir.grid(column=0,row=5, sticky='NEWS')
        self.label_error_viewing_dir_value = ttk.Label(self.panel_algo, anchor='w', text="0.0%", font=font.Font(size=16))
        self.label_error_viewing_dir_value.grid(column=1,row=5, sticky='NEWS', padx=5)
        self.label_swarm_center = ttk.Label(self.panel_algo, anchor='w', text="Swarm center: ", font=font.Font(size=14))
        self.label_swarm_center.grid(column=0,row=6, sticky='NEWS')
        self.label_swarm_center_value = ttk.Label(self.panel_algo, anchor='w', text="[0.0;0.0;0.0]", font=font.Font(size=16))
        self.label_swarm_center_value.grid(column=1,row=6, sticky='NEWS', padx=5)

        # Coverage metric
        self.label_coverage = ttk.Label(self.panel_algo, anchor='w', text="Coverage: ", font=font.Font(size=14))
        self.label_coverage.grid(column=0,row=7, sticky='NEWS')
        self.label_coverage_value = ttk.Label(self.panel_algo, anchor='w', text="0.00%", font=font.Font(size=16))
        self.label_coverage_value.grid(column=1,row=7, sticky='NEWS', padx=5)

        # Output CSV file
        self.label_output = ttk.Label(self.panel_algo, anchor='w', text="Output data filename: ", font=font.Font(size=12))
        self.label_output.grid(column=0,row=8, sticky='NEWS')
        self.panel_output_file = tk.Frame(self.panel_algo)
        self.panel_output_file.grid_columnconfigure(0, weight=2)
        self.panel_output_file.grid_columnconfigure((1,2), weight=1)
        self.panel_output_file.grid_rowconfigure(0, weight=1)
        self.panel_output_file.grid(row=8, column=1, sticky='NEWS')
        self.textbox_output_csv = ttk.Entry(self.panel_output_file, width=30, font=font.Font(size=12), textvariable=self.var_output_csv)
        self.textbox_output_csv.grid(column=0,row=0, sticky='W', padx=5)
        self.button_start_recording = ttk.Button(self.panel_output_file, text="Start", command=self.start_recording_callback, state='disabled')
        self.button_start_recording.grid(column=1,row=0, sticky='EW', padx=5)
        self.button_stop_recording = ttk.Button(self.panel_output_file, text="Stop", command=self.stop_recording_callback, state='disabled')
        self.button_stop_recording.grid(column=2,row=0, sticky='EW', padx=5)

        # Autoun features
        self.label_autorun_sim = ttk.Label(self.panel_algo, anchor='w', text="Autorun simulations: ", font=font.Font(size=14))
        self.label_autorun_sim.grid(column=0,row=9, sticky='NEWS')
        self.textbox_autorun_filename = ttk.Entry(self.panel_algo, width=30, font=font.Font(size=12), textvariable=self.var_autorun_filename)
        self.textbox_autorun_filename.grid(column=1,row=9, sticky='EW', padx=5)
        self.btn_autorun_sim = ttk.Button(self.panel_algo, text="RUN", command=self._btn_autorun_callback)
        self.btn_autorun_sim.grid(column=2,row=9, sticky='NEWS', padx=0)

    def init_sidebar_components(self):
        # Title
        self.label_title = ttk.Label(self.panel_title, anchor='center', text="DRONE SWARM BOUNDARIES DETECTION", justify='center',
                                     font=font.Font(name='Helvetica', weight='bold', size=16), foreground='white', background=self.panel_title['bg'])
        self.label_title.grid(column=0,row=0, sticky='NEWS')

        # Drone number
        self.label_drone_nb = ttk.Label(self.panel_params, anchor='w', text="# drones: ")
        self.label_drone_nb.grid(column=0,row=0, sticky='NEWS', pady=5)
        self.spinner_drone_nb = ttk.Spinbox(self.panel_params, increment=1,from_=1, to=50, command=lambda: self.var_drone_count.set(self.spinner_drone_nb.get()))
        self.spinner_drone_nb.bind("<Return>", lambda e: self.var_drone_count.set(self.spinner_drone_nb.get()))
        self.spinner_drone_nb.grid(row=0,column=1,sticky='w', padx=5, pady=5)
        self.spinner_drone_nb.set(self.var_drone_count.get())

        # Neighbors
        self.panel_neighbors = tk.Frame(self.panel_params)
        self.panel_neighbors.grid(column=0, row=1, columnspan=3, rowspan=1, sticky='NWES', padx=0, pady=5)
        self.panel_neighbors.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        self.panel_neighbors.grid_rowconfigure(0, weight=1)
        self.label_neighbors = ttk.Label(self.panel_neighbors, anchor='w', text="Neighbors: ")
        self.label_neighbors.grid(column=0,row=0, sticky='NEWS')
        self.listbox_neighbors_algo = ttk.Combobox(self.panel_neighbors, values=["Eucledian", "Topological", "Voronoi", "VLOS"])
        self.listbox_neighbors_algo.set(self.app_config['neighbors'].get('metric', 'Topological'))
        self.listbox_neighbors_algo.grid(row=0,column=1,sticky='we', padx=5)
        self.listbox_neighbors_algo.bind("<<ComboboxSelected>>", lambda e: self.var_neighbors_metric.set(self.listbox_neighbors_algo.get()))
        self.label_neighbors_algo_param = ttk.Label(self.panel_neighbors, anchor='e', text="# ")
        self.label_neighbors_algo_param.grid(column=2,row=0,sticky='NEWS')
        self.spinner_neighbors = ttk.Spinbox(self.panel_neighbors, increment=1,from_=0, to=self.var_drone_count.get()-1, textvariable=self.var_neighbors_count)
        self.spinner_neighbors.grid(row=0,column=3,sticky='w', padx=5)
        self.label_spinner_r_agent = ttk.Label(self.panel_neighbors, anchor='e', text="r_agent: ")
        self.spinner_r_agent = ttk.Spinbox(self.panel_neighbors, increment=0.01, from_=0, to=1, textvariable=self.var_neighbors_r_agent)
        # Disable all components in the neighbors panel
        for child in self.panel_neighbors.winfo_children():
            child.config(state='disabled')

        # Control params panel
        self.panel_control_scheme = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
        self.panel_control_scheme.grid(column=0, row=2, columnspan=3, rowspan=1, sticky='NWES', padx=0, pady=0)
        self.panel_control_scheme.grid_columnconfigure((0,1,2), weight=1)
        self.panel_control_scheme.grid_rowconfigure(0, weight=1)
        # COntrol scheme
        self.label_control_scheme = ttk.Label(self.panel_control_scheme, anchor='w', text="Control scheme: ")
        self.label_control_scheme.grid(column=0,row=0, sticky='NEWS')
        self.listbox_control_scheme = ttk.Combobox(self.panel_control_scheme, values=["Olfati-Saber", "Reynolds"])
        self.listbox_control_scheme.set(self.var_swarming_algorithm.get())
        self.listbox_control_scheme.grid(row=0,column=1, sticky='we', padx=10)
        self.listbox_control_scheme.bind("<<ComboboxSelected>>", lambda e: self.var_swarming_algorithm.set(self.listbox_control_scheme.get()))
        self.button_control_scheme_params = ttk.Button(self.panel_control_scheme, text="Edit Params", command=self._btn_control_scheme_params_callback)
        self.button_control_scheme_params.grid(row=0,column=2, sticky='EW', padx=10, ipady=5)
        
        # Noise panel
        self.pnael_noise = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
        self.pnael_noise.grid(column=0, row=3, columnspan=3, rowspan=2, sticky='NWES', padx=0, pady=0)
        self.pnael_noise.grid_columnconfigure((0,1,2,3,4,5,6), weight=1)
        self.pnael_noise.grid_rowconfigure((0,1), weight=1)
        self.label_noise = ttk.Label(self.pnael_noise, anchor='w', text="Noise: ")
        self.label_noise.grid(column=0,row=0, sticky='NEWS')
        self.listbox_noise_type = ttk.Combobox(self.pnael_noise, values=["None", "Uniform", "Gaussian"])
        self.listbox_noise_type.set(self.app_config['noise'].get('type', 'None'))
        self.listbox_noise_type.grid(column=1,row=0, sticky='we', padx=5)
        self.listbox_noise_type.bind("<<ComboboxSelected>>", lambda e: self.var_noise_type.set(self.listbox_noise_type.get()))
        self.btn_apply_noise_all = ttk.Button(self.pnael_noise, text="Apply to all", state="disabled", command=self.btn_apply_all_callback)
        self.btn_apply_noise_all.grid(column=3,row=0, columnspan=2, sticky='EW', ipady=5)
        self.label_noise_pos = ttk.Label(self.pnael_noise, anchor='e', text="Dist:", justify='right')
        self.label_noise_pos.grid(column=0,row=1, sticky='NEWS')
        self.spinner_noise_pos = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_param_dist)
        self.spinner_noise_pos.grid(column=1, row=1, sticky='W', padx=5)
        self.label_noise_heading = ttk.Label(self.pnael_noise, anchor='w', text="Dir (sensing cone): ", justify='right')
        self.label_noise_heading.grid(column=2,row=1, sticky='NEWS')
        self.spinner_noise_dir = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_param_dir)
        self.spinner_noise_dir.grid(column=3, row=1, sticky='W', padx=5)
        self.label_noise_heading = ttk.Label(self.pnael_noise, anchor='w', text="Heading: ", justify='right')
        self.label_noise_heading.grid(column=4,row=1, sticky='NEWS')
        self.spinner_noise_heading = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_param_heading)
        self.spinner_noise_heading.grid(column=5, row=1, sticky='W', padx=5)
        
        # Target panel
        self.panel_target = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
        self.panel_target.grid(column=0, row=5, columnspan=3, rowspan=1, sticky='NWES', padx=0, pady=0)
        self.panel_target.grid_columnconfigure((0,1,2,4), weight=1)
        self.panel_target.grid_rowconfigure((1,2,3), weight=1)
        self.label_swarm_motion = ttk.Label(self.panel_target, anchor='w', text="Swarm motion", justify='left', font=font.Font(size=11, underline=True))
        self.label_swarm_motion.grid(column=0,row=0, sticky='NEWS', padx=5, pady=5)
        self.radio_target = ttk.Radiobutton(self.panel_target, text="Target", variable=self.var_trajectory_mode, value='target' ,command=self._update_swarm_target)
        self.radio_target.grid(column=0,row=1, sticky='NEWS', padx=5)
        self.radio_trajectory = ttk.Radiobutton(self.panel_target, text="Trajectory", variable=self.var_trajectory_mode, value='trajectory', command=self._update_swarm_target)
        self.radio_trajectory.grid(column=0,row=2, sticky='NEWS', padx=5)
        self.radio_keyboard = ttk.Radiobutton(self.panel_target, text="Keyboard", variable=self.var_trajectory_mode, value='keyboard', command=self._update_swarm_target)
        self.radio_keyboard.grid(column=0,row=3, sticky='NEWS', padx=5)
        self.textbox_target = ttk.Entry(self.panel_target, text='', font=font.Font(size=10))
        self.textbox_target.bind("<Return>", self._update_swarm_target)
        self.textbox_target.grid(column=1, row=1, sticky='WE', padx=5)
        self.label_target_format = ttk.Label(self.panel_target, anchor='w', text="#.#;#.#;#.# or empty", justify='left', font=font.Font(size=10))
        self.label_target_format.grid(column=2,row=1, sticky='NEWS')
        self.btn_update_target = ttk.Button(self.panel_target, text="Update", command=self._update_swarm_target)
        self.btn_update_target.grid(column=4,row=1, sticky='EW', ipady=2, padx=5)
        self.listbox_trajectories = ttk.Combobox(self.panel_target, values=["circle","figure8"], state='disabled')
        self.listbox_trajectories.set(self.app_config['simulation'].get('trajectory_type', 'circle'))
        self.listbox_trajectories.grid(column=1,row=2, sticky='WE', padx=5)

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
        self.btn_center = ttk.Button(self.panel_sim, text="Center view on swarm", command=self._btn_center_callback, state='disabled')
        self.btn_center.grid(column=1, row=2, sticky='EW', padx=10, ipady=10)
        text_btn_rendering = "Rendering: ON" if self.render_env else "Rendering: OFF"
        self.btn_rendering = ttk.Button(self.panel_sim, text=text_btn_rendering, command=self._btn_rendering_callback)
        self.btn_rendering.grid(column=2, row=1, sticky='EW', padx=10, pady=5)

        # 3D quick view selection panel (top, sie, front back, reset)
        self.panel_3D_quick_views = tk.Frame(self.panel_sim)
        self.panel_3D_quick_views.grid_columnconfigure((0,1,2), weight=1)
        self.panel_3D_quick_views.grid(row=2, column=2, sticky='NWES')
        self.btn_3D_top_view = ttk.Button(self.panel_3D_quick_views, text="TOP", command=lambda: self.renderer.set_view('top'))
        self.btn_3D_top_view.grid(column=1,row=1, sticky='NEWS', padx=5)
        self.btn_3D_left_view = ttk.Button(self.panel_3D_quick_views, text="LEFT", command=lambda: self.renderer.set_view('left'))
        self.btn_3D_left_view.grid(column=0,row=1, sticky='NEWS', padx=5)
        self.btn_3D_right_view = ttk.Button(self.panel_3D_quick_views, text="RIGHT", command=lambda: self.renderer.set_view('right'))
        self.btn_3D_right_view.grid(column=2,row=1, sticky='NEWS', padx=5)
        self.btn_3D_front_view = ttk.Button(self.panel_3D_quick_views, text="FRONT", command=lambda: self.renderer.set_view('front'))
        self.btn_3D_front_view.grid(column=1,row=2, sticky='NEWS', padx=5)
        self.btn_3D_back_view = ttk.Button(self.panel_3D_quick_views, text="BACK", command=lambda: self.renderer.set_view('back'))
        self.btn_3D_back_view.grid(column=1,row=0, sticky='NEWS', padx=5)
        self.btn_3D_reset_view = ttk.Button(self.panel_3D_quick_views, text="RESET", command=self._btn_reset_view_callback)
        self.btn_3D_reset_view.grid(column=2,row=2, sticky='NEWS', padx=5)

        # Save app params button
        self.btn_save_params = ttk.Button(self.panel_sim, text="Save configuration", command=self.export_app_config)
        self.btn_save_params.grid(column=0,row=2, sticky='EW', padx=10, ipady=10)

        # Simulation timestep
        self.panel_sim_timestep = tk.Frame(self.panel_sim)
        self.panel_sim_timestep.grid(column=0, row=3, columnspan=2, sticky='NWES')
        self.panel_sim_timestep.grid_columnconfigure(0, weight=2)
        self.panel_sim_timestep.grid_columnconfigure(2, weight=1)
        self.panel_sim_timestep.grid_rowconfigure(0, weight=1)
        self.label_sim_dt = ttk.Label(self.panel_sim_timestep, text="dt: ", justify='right', font=font.Font(size=11))
        self.label_sim_dt.grid(column=1, row=0, sticky='W')
        self.spinner_sim_dt = ttk.Spinbox(self.panel_sim_timestep, increment=0.001, from_=0.001, to=1, textvariable=self.var_sim_dt, width=10)
        self.spinner_sim_dt.grid(column=2, row=0, sticky='W', padx=5)
        self.label_sim_total_time = ttk.Label(self.panel_sim_timestep, text='Simulation time: 0.000 s', justify='left', font=font.Font(size=10))
        self.label_sim_total_time.grid(column=0, row=0, sticky='NEWS', padx=5)

        # Neihbors display
        self.panel_sim_neighbors = tk.Frame(self.panel_sim)
        # self.panel_sim_neighbors.grid_columnconfigure(0, weight=1)
        self.panel_sim_neighbors.grid_columnconfigure(2, weight=1)
        self.panel_sim_neighbors.grid_rowconfigure(0, weight=1)
        self.panel_sim_neighbors.grid(row=3, column=2, columnspan=1, sticky='NWES')
        self.label_neighbor_sampling = ttk.Label(self.panel_sim_neighbors, text="Neighbor sampling: ", justify='right')
        self.label_neighbor_sampling.grid(column=0, row=0, sticky='E')
        self.spinner_neighbor_sampling = ttk.Spinbox(self.panel_sim_neighbors, increment=1, from_=1, to=100, textvariable=self.var_neighbors_sampling, width=10)
        self.spinner_neighbor_sampling.grid(column=1, row=0, sticky='W', padx=5)
        self.listbox_neighbors_select = ttk.Combobox(self.panel_sim_neighbors, values=["None", "Selected", "All"], width=10)
        self.listbox_neighbors_select.set(self.app_config['neighbors'].get('computation', 'None'))
        self.listbox_neighbors_select.bind("<<ComboboxSelected>>", lambda e: self._set_neighbors_algo_params())
        self.listbox_neighbors_select.grid(column=2, row=0, sticky='W', padx=15)


    def init_env_components(self):
        # Obstacle parameters
        self.label_obstacle_params = ttk.Label(self.panel_env, text='Obstacle Parameters', background='lightgray', font=font.Font(size=12))
        self.label_obstacle_params.grid(row=0, column=0, sticky='W', pady=5, padx=10)

        # Obstacle panel
        self.panel_obstacle_params = tk.Frame(self.panel_env, borderwidth=2, relief='ridge')
        self.panel_obstacle_params.columnconfigure(0, weight=1)
        self.panel_obstacle_params.rowconfigure(7, weight=1)
        self.panel_obstacle_params.grid(row=1, column=0, sticky='NWES', padx=10)

        # Treeview for current obstacles
        self.treeview_obstacles = ttk.Treeview(self.panel_obstacle_params, columns=('type', 'center', 'height', 'radius'), show='tree headings')
        self.treeview_obstacles.column('#0', width=30)
        self.treeview_obstacles.heading('type', text='Type')
        self.treeview_obstacles.column('type', anchor='center')
        self.treeview_obstacles.heading('center', text='Center')
        self.treeview_obstacles.column('center', anchor='center')
        self.treeview_obstacles.heading('height', text='Height')
        self.treeview_obstacles.column('height', width=60, anchor='center')
        self.treeview_obstacles.heading('radius', text='Radius')
        self.treeview_obstacles.column('radius', width=60, anchor='center')
        self.treeview_obstacles.grid(row=0, column=0, rowspan=9, sticky='NWES', padx=0)
        self.treeview_obstacles.configure(selectmode='browse')
        self.treeview_obstacles.bind("<ButtonRelease-1>", self._select_obstacle_callback)

        # X coordinate
        self.label_obstacle_x = ttk.Label(self.panel_obstacle_params, text='Center X:')
        self.label_obstacle_x.grid(row=1, column=1, sticky='W', padx=10)
        self.spinner_obstacle_x = ttk.Spinbox(self.panel_obstacle_params, increment=0.1, from_=-np.inf, to=np.inf, command=self._update_obstacle_params)
        self.spinner_obstacle_x.grid(row=1, column=2, sticky='W')
        self.spinner_obstacle_x.bind("<Return>", self._update_obstacle_params)
        self.spinner_obstacle_x.insert(0, 0.0)

        # Y coordinate
        self.label_obstacle_y = ttk.Label(self.panel_obstacle_params, text='Center Y:')
        self.label_obstacle_y.grid(row=2, column=1, sticky='W', padx=10)
        self.spinner_obstacle_y = ttk.Spinbox(self.panel_obstacle_params, increment=0.1, from_=-np.inf, to=np.inf, command=self._update_obstacle_params)
        self.spinner_obstacle_y.grid(row=2, column=2, sticky='W')
        self.spinner_obstacle_y.bind("<Return>", self._update_obstacle_params)
        self.spinner_obstacle_y.insert(0, 0.0)

        # Z coordinate
        self.label_obstacle_z = ttk.Label(self.panel_obstacle_params, text='Center Z:')
        self.label_obstacle_z.grid(row=3, column=1, sticky='W', padx=10)
        self.spinner_obstacle_z = ttk.Spinbox(self.panel_obstacle_params, increment=0.1, from_=-np.inf, to=np.inf, command=self._update_obstacle_params)
        self.spinner_obstacle_z.grid(row=3, column=2, sticky='W')
        self.spinner_obstacle_z.bind("<Return>", self._update_obstacle_params)
        self.spinner_obstacle_z.insert(0, 0.0)

        # Height
        self.label_obstacle_height = ttk.Label(self.panel_obstacle_params, text='Height:')
        self.label_obstacle_height.grid(row=4, column=1, sticky='W', padx=10)
        self.spinner_obstacle_height = ttk.Spinbox(self.panel_obstacle_params, increment=0.1, from_=0, to=np.inf, command=self._update_obstacle_params)
        self.spinner_obstacle_height.grid(row=4, column=2, sticky='W')
        self.spinner_obstacle_height.bind("<Return>", self._update_obstacle_params)
        self.spinner_obstacle_height.insert(0, 10.0)

        # Radius
        self.label_obstacle_radius = ttk.Label(self.panel_obstacle_params, text='Radius:')
        self.label_obstacle_radius.grid(row=5, column=1, sticky='W', padx=10)
        self.spinner_obstacle_radius = ttk.Spinbox(self.panel_obstacle_params, increment=0.1, from_=0, to=np.inf, command=self._update_obstacle_params)
        self.spinner_obstacle_radius.grid(row=5, column=2, sticky='W')
        self.spinner_obstacle_radius.bind("<Return>", self._update_obstacle_params)
        self.spinner_obstacle_radius.insert(0, 1.0)

        # buttons panel
        self.panel_obstacle_buttons = tk.Frame(self.panel_obstacle_params)
        self.panel_obstacle_buttons.grid(row=6, column=1, columnspan=2, sticky='NES', padx=0, pady=5)
        self.panel_obstacle_buttons.rowconfigure(0, weight=1)

        # Add obstacle button
        self.btn_add_obstacle = tk.Button(self.panel_obstacle_buttons, text='+', font=font.Font(size=14, weight='bold'), command=self._add_obstacle_callback)
        self.btn_add_obstacle.grid(row=0, column=0, sticky='WE', padx=5)

        # Remove obstacle button
        self.btn_remove_obstacle = tk.Button(self.panel_obstacle_buttons, font=font.Font(size=14, weight='bold'),text='-', command=self._remove_obstacle_callback)
        self.btn_remove_obstacle.grid(row=0, column=1, sticky='WE', padx=5)
        self.btn_remove_obstacle.config(state='disabled')

        # Save environment
        self.btn_save_env = tk.Button(self.panel_obstacle_params, text='Save environment', command=self._save_env_callback)
        self.btn_save_env.grid(row=8, column=1, columnspan=2, sticky='WE', padx=5, pady=5)

    def init_concave_hull_components(self):
        
        # Add a heading label
        self.label_concave_hull = ttk.Label(self.panel_hull, anchor='w', text="Concave Hull", justify='center', font=font.Font(size=14, weight='bold'), background='lightgray')
        self.label_concave_hull.grid(column=0, row=0, ipady=10, ipadx=10, sticky='NEWS')

        # Add the alpha label and entry box panel directly under the heading
        self.panel_alpha = tk.Frame(self.panel_hull)
        self.panel_alpha.grid(column=0, row=1, sticky='W', padx=10, pady=10)

        # Add the label for alpha
        self.label_alpha = ttk.Label(self.panel_alpha, text="Alpha: ", font=font.Font(size=12))
        self.label_alpha.grid(column=0, row=0, sticky='E')

        # Add the entry box for alpha
        self.var_alpha = tk.DoubleVar(value=1.0)
        self.spinner_alpha = ttk.Spinbox(self.panel_alpha, textvariable=self.var_alpha, width=8, from_=0.0, to=1.0, increment=0.05)  # Smaller width
        self.spinner_alpha.grid(column=1, row=0, sticky='W')

        # Add a button to toggle concave hull computation and rendering
        self.btn_toggle_hull = ttk.Button(self.panel_hull, text="Enable Concave Hull", command=self._btn_concave_hull_callback)
        self.btn_toggle_hull.grid(column=0, row=2, ipady=10, padx=10, sticky='EW')

        # Trace the alpha for real-time updates
        self.var_alpha.trace_add('write', self._alpha_changed_callback)



    #==================================#
    #   SIM COMPONENTS CALLBACK        #
    #==================================# 
        
    def _initialize_simulation(self, verbose=True):
        # Retrieve all necessary parameters from app widgets
        nb_drones = self.var_drone_count.get()
        target_text = self.textbox_target.get()
        self.env.set_target(target_text)

        self.swarm = Swarm(self.env, count=nb_drones, area=self.spawn_area,
                           is_2d=self.swarm_2d, trajectory=self.swarm_traj)
        dt = float(self.var_sim_dt.get())
        self._recorder._swarm = self.swarm
        self.sim = Simulator(dt, self.swarm, self._recorder)
        #self.swarm.initialize_random_vel([0.1, 0.5, -0.3, 0.3, 0, 0.2])
        self._set_neighbors_algo_params()
        self._set_swarm_algo_params()
        self.swarm.initialize_members()
        self.swarm.set_pd_controller(**self.app_config['simulation']['pd_controller'])
        self.swarm.compute_neighborhood()
        if verbose:
            print("Initializing swarm with parameters: ")
            print(self.swarm.algo_params)
            self.swarm.print_swarm()
        #self.swarm.migration_point = np.array([5,0,10])
        if self.render_env:
            self.renderer._swarm_ref = self.swarm
            self.renderer.start()
        #self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,0.2]))
        # Unlock buttons
        if self.render_env:
            self.btn_2D_view.config(state='normal')
            self.btn_center.config(state='normal')
        self.btn_simulate.config(state='normal')
        # Enable all components in the neighbors panel
        for child in self.panel_neighbors.winfo_children():
            child.config(state='normal')
        self._update_neighbors_panel_components()
        self.noise_changed_callback(verbose=verbose)
        self.swarm.set_viewing_algorithm(self.var_viewing_metric_algorithm.get())
        self.listbox_viewing_algo.configure(state='normal')
        self.radio_2D_viewing.config(state='normal')
        self.radio_3D_viewing.config(state='normal')
        self.button_start_recording.config(state='normal')
        self.btn_apply_noise_all.config(state='normal')
        if self.swarm.is_2D:
            self.var_viewing_metric_dim.set(2)
            self.radio_3D_viewing.config(state='disabled')
        self.swarm.set_migration_mode(self.var_trajectory_mode.get())
        self.viewing_metric_changed_callback(None)
        
    def _set_swarm_algo_params(self, *args):
        if self.swarm is None:
            return

        try:
            self.swarm.algo_params.update( {
                'algorithm': self.var_swarming_algorithm.get(),
                **self.app_config['swarming_algorithm'].get('params', {})
            })
        except ValueError as e:
            pass
        except Exception as e:
            print("Error setting swarm algo params: {0}".format(e))

    def _update_swarm_target(self, *args):
        mode = self.var_trajectory_mode.get()
        match mode:
            case 'target':
                self.textbox_target.config(state='normal')
                self.btn_update_target.config(state='normal')
                self.listbox_trajectories.config(state='disabled')
                try:
                    self.env.set_target(self.textbox_target.get())
                except ValueError as e:
                    pass
            case 'trajectory':
                self.env.target = None
                self.textbox_target.config(state='disabled')
                self.btn_update_target.config(state='disabled')
                self.listbox_trajectories.config(state='normal')
            case 'keyboard':
                self.env.target = None
                self.textbox_target.config(state='disabled')
                self.btn_update_target.config(state='disabled')
                self.listbox_trajectories.config(state='disabled')
        if self.swarm is not None:
            self.swarm.set_migration_mode(mode)
                

    def _update_neighbors_panel_components(self, *args):
        current_algo = self.var_neighbors_metric.get()
        if self.listbox_neighbors_algo.get() != current_algo:
            self.listbox_neighbors_algo.set(current_algo)
        self.spinner_neighbors.config(state='normal')
        if self.label_spinner_r_agent in self.panel_neighbors.winfo_children():
            self.label_spinner_r_agent.grid_forget()
            self.spinner_r_agent.grid_forget()
        match current_algo.upper():
            case 'EUCLEDIAN':
                self.label_neighbors_algo_param.config(text="radius")
                default_val = self.app_config['neighbors'].get('sensing_range', 1.0)
                current_val = self.swarm.get_neighbor_metric('sensing_range', default_val)
                self.var_neighbors_sensing_range.set(current_val)
                self.spinner_neighbors.config(from_=0, to=100, increment=0.1, textvariable=self.var_neighbors_sensing_range)
            case 'TOPOLOGICAL':
                self.label_neighbors_algo_param.config(text="# ")
                default_val = self.app_config['neighbors'].get('count', 1)
                current_val = self.swarm.get_neighbor_metric('count', default_val)
                self.var_neighbors_count.set(current_val)
                self.spinner_neighbors.config(from_=0, to=self.var_drone_count.get()-1, increment=1, textvariable=self.var_neighbors_count)
            case 'VORONOI':
                self.label_neighbors_algo_param.config(text="")
                self.spinner_neighbors.config(state='disabled')
            case 'VLOS':
                self.label_neighbors_algo_param.config(text="radius ")
                default_val = self.app_config['neighbors'].get('sensing_range', 1.0)
                self.var_neighbors_sensing_range.set(self.swarm.get_neighbor_metric('sensing_range', default_val))
                self.spinner_neighbors.config(from_=0, to=100, increment=0.1, textvariable=self.var_neighbors_sensing_range)
                self.label_spinner_r_agent.grid(column=4,row=0,sticky='NEWS')
                self.spinner_r_agent.grid(column=5,row=0,sticky='W')
        self._set_neighbors_algo_params()
    
    def _set_neighbors_algo_params(self, *args):
        if self.swarm is None:
            return
        try:
            self.swarm.update_neighbors_metric({
                'computation': self.listbox_neighbors_select.get(),
                'metric': self.listbox_neighbors_algo.get(),
                'sampling': self.var_neighbors_sampling.get(),
                'count' : self.var_neighbors_count.get(),
                'sensing_range': self.var_neighbors_sensing_range.get(),
                'r_agent': self.var_neighbors_r_agent.get()
            })
            self.swarm.member_size = self.var_neighbors_r_agent.get()
            if self.sim.paused():
                self.swarm.compute_neighborhood()
        except Exception as e:
            print("Error setting neighbors algo params: {0}".format(e))

    def noise_changed_callback(self, verbose=True, *args):
        if self.listbox_noise_type.get() != self.var_noise_type.get():
            self.listbox_noise_type.set(self.var_noise_type.get())
        if self.var_noise_type.get().upper() == 'NONE':
            self.spinner_noise_pos.config(state='disabled')
            self.spinner_noise_dir.config(state='disabled')
            self.spinner_noise_heading.config(state='disabled')
        else:
            self.spinner_noise_pos.config(state='normal')
            self.spinner_noise_dir.config(state='normal')
            self.spinner_noise_heading.config(state='normal')
        if self.swarm is None:
            return
        try:
            self.swarm.set_noise(self.listbox_noise_type.get(), self.var_noise_param_dist.get(), self.var_noise_param_dir.get(), self.var_noise_param_heading.get())
            if verbose:
                print('Noise parameters changed: {0}'.format(self.swarm.get_noise()))
        except Exception as e:
            print("Error setting noise: {0}".format(e))

    def drone_selection_changed(self, *args):
        # Update neighbors
        self.swarm.compute_neighborhood()
        # Query noise parameters
        noise = self.swarm.get_noise()
        self.listbox_noise_type.set(noise.get('type', 'None'))
        self.var_noise_param_dist.set(noise.get('param_dist', self.app_config['noise'].get('param_dist', 0.0)))
        self.var_noise_param_dir.set(noise.get('param_dir', self.app_config['noise'].get('param_dir', 0.0)))
        self.var_noise_param_heading.set(noise.get('param_heading', self.app_config['noise'].get('param_heading', 0.0)))

    def swarming_algo_changed(self, *args):
        if self.listbox_control_scheme.get() != self.var_swarming_algorithm.get():
            self.listbox_control_scheme.set(self.var_swarming_algorithm.get())
        if self.swarm is None:
            return
        self.swarm.set_swarming_algorithm(self.var_swarming_algorithm.get())

    def viewing_metric_changed_callback(self, *args):
        # Viewing algorithm
        algo = self.var_viewing_metric_algorithm.get()
        self.listbox_viewing_algo.set(algo)
        self.listbox_convex_hull_faces.set(self.var_viewing_metric_faces.get())
        if algo.upper() == 'outer':
            self.panel_algo_params_outer.grid(column=1, row=1, sticky='NWES', padx=5, pady=0)
        else:
            self.panel_algo_params_outer.grid_forget()
        if algo.upper() == 'CONVEX_HULL':
            self.panel_algo_params_convex_hull.grid(column=1, row=1, sticky='NWES', padx=5, pady=0)
        else:
            self.panel_algo_params_convex_hull.grid_forget()
        # Other params
        dim = self.var_viewing_metric_dim.get()
        params = {'nb_points': self.var_viewing_metric_outer_points.get(), 
                  "faces": self.var_viewing_metric_faces.get(),
                  "in_2d": dim==2}
        self.swarm.set_viewing_algorithm(algo, params)
        if dim == 2:
            self.label_coverage['text'] = "Coverage (circle): "
        else:
            self.label_coverage['text'] = "Coverage (sphere): "

    def update_sim_data(self, sim_time, **kwargs):
        # Viewing dir metrics
        selected_drone = self.swarm.selected_drone
        viewing_dir = self.swarm.members[selected_drone].estimated_viewing_dir
        true_dir = self.swarm.members[selected_drone].ground_truth_viewing_dir
        error = self.swarm.members[selected_drone].viewing_error
        self.label_estimated_viewing_dir_value.config(text=f'[{viewing_dir[0]:.3f};{viewing_dir[1]:.3f};{viewing_dir[2]:.3f}]')
        self.label_true_viewing_dir_value.config(text=f'[{true_dir[0]:.3f};{true_dir[1]:.3f};{true_dir[2]:.3f}]')
        self.label_error_viewing_dir_value.config(text=f'{error:.2f} °')
        self.label_swarm_center_value.config(text=f'[{self.swarm.swarm_center[0]:.2f};{self.swarm.swarm_center[1]:.2f};{self.swarm.swarm_center[2]:.2f}]')
        # Simulation time
        self.label_sim_total_time.configure(text=f'Simulation time: {sim_time:.3f} s')
        # Coverage
        self.label_coverage_value.config(text=f'{self.swarm.swarm_coverage*100:.2f}%')

    def _update_swarm_count(self, *args):
        if self.spinner_drone_nb.get() != self.var_drone_count.get():
            self.spinner_drone_nb.set(self.var_drone_count.get())
        self.spinner_neighbors.config(to=self.var_drone_count.get()-1)
        if self.var_neighbors_count.get() >= self.var_drone_count.get():
            self.var_neighbors_count.set(self.var_drone_count.get()-1)
        if self.swarm is not None:
            if not self.sim.paused() and self.var_drone_count.get() < self.swarm.count:
                # Pause simulation
                self._btn_pause_callback()
                time.sleep(self.sim._dt)
            self.swarm.set_count(self.var_drone_count.get())

    def _update_sim_dt(self, *args):
        if self.spinner_sim_dt.get() != self.var_sim_dt.get():
            self.spinner_sim_dt.set(self.var_sim_dt.get())
        if self.sim is not None:
            self.sim._dt = float(self.var_sim_dt.get())
        if self._recorder is not None:
            self._recorder._dt = float(self.var_sim_dt.get())

    def _generate_obstacles_list(self):
        """
        Generate a list of obstacles for the treeview widget
        """
        # Clear all items
        for item in self.treeview_obstacles.get_children():
            self.treeview_obstacles.delete(item)
        obstacles = self.env.obstacles
        # Add new items with index
        for idx, obs in enumerate(obstacles):
            self.treeview_obstacles.insert('', 'end', text=idx, values=(obs.__class__(), obs.center, obs.height, obs.radius))
        self.btn_remove_obstacle.config(state='disabled')

    def _select_obstacle_callback(self, event):
        """
        Handles the selection of an obstacle in the treeview widget
        Args:
            event (_type_): _description_

        """
        item = self.treeview_obstacles.selection()[0]
        idx = self.treeview_obstacles.item(item, 'text')
        obs = self.env.obstacles[idx]
        self.env.select_obstacle(idx)
        self.spinner_obstacle_x.set(obs.center[0])
        self.spinner_obstacle_y.set(obs.center[1])
        self.spinner_obstacle_z.set(obs.center[2])
        self.spinner_obstacle_height.set(obs.height)
        self.spinner_obstacle_radius.set(obs.radius)
        self.btn_remove_obstacle.config(state='normal')

    def obstacle_moved_callback(self, *args):
        if self.env.get_selected_obstacle() is None:
            return
        obs = self.env.get_selected_obstacle()
        self.spinner_obstacle_x.set(obs.center[0])
        self.spinner_obstacle_y.set(obs.center[1])
        self.spinner_obstacle_z.set(obs.center[2])
        self.spinner_obstacle_height.set(obs.height)
        self.spinner_obstacle_radius.set(obs.radius)
        self.btn_remove_obstacle.config(state='normal')

    def obstacle_clicked_callback(self, *args):
        if self.env.get_selected_obstacle() is None:
            return
        idx = self.env.get_selected_obstacle_idx()
        self.treeview_obstacles.selection_set(self.treeview_obstacles.get_children()[idx])
        self.obstacle_moved_callback()
        if self.tabbed_pane.index(self.tabbed_pane.select()) != 1:
            self.tabbed_pane.select(1)

    def _update_obstacle_params(self, *args):
        """
        Updates the obstacle parameters based on the values in the spinbox widgets
        """
        items = self.treeview_obstacles.selection()
        if len(items) == 0:
            return
        item = items[0]
        idx = self.treeview_obstacles.item(item, 'text')
        obs = self.env.obstacles[idx]
        try:
            x = float(self.spinner_obstacle_x.get())
            y = float(self.spinner_obstacle_y.get())
            z = float(self.spinner_obstacle_z.get())
            height = float(self.spinner_obstacle_height.get())
            radius = float(self.spinner_obstacle_radius.get())
        except ValueError as e:
            return
        obs.center = np.array([x, y, z])
        obs.height = height
        obs.radius = radius
        # update treeview
        self.treeview_obstacles.item(item, values=(obs.__class__(), obs.center, obs.height, obs.radius))

    def _tabbed_panel_changed(self, event):
        """
        Handles the event when the user changes the tab in the notebook widget
        """
        if self.tabbed_pane.index(self.tabbed_pane.select()) == 1:
            return
        self.env.deselect_obstacle()
        self.treeview_obstacles.selection_remove(self.treeview_obstacles.selection())

    def _alpha_changed_callback(self, *args):
        if self.swarm and self.swarm.concave_hull_enabled:
            try:
                alpha = self.var_alpha.get()
            except (ValueError, tk.TclError):
                alpha = 1.0  # Default value
            self.swarm.set_alpha(alpha) 


    #==================================#
    #          BUTTON CALLBACKS        #
    #==================================#

    def _button_reset_callback(self):
        self.label_sim_total_time.config(foreground='black', text='Simulation time: 0.000 s')
        if self.render_env:
            self.renderer.reset()
            self.renderer._swarm_ref = None
        # Disabling simulation controls
        self.btn_center.config(state='disabled')
        self.btn_simulate.config(state='disabled')
        self.btn_2D_view.config(state='disabled')
        self.radio_2D_viewing.config(state='disabled')
        self.radio_3D_viewing.config(state='disabled')
        self.button_stop_recording.config(state='disabled')
        self.button_start_recording.config(state='disabled')
        self.btn_apply_noise_all.config(state='disabled')
        self.btn_pause.config(state='disabled')
        self.swarm = None
        self.update_frontend_running = False
        # Disable all components in the neighbors panel
        for child in self.panel_neighbors.winfo_children():
            child.config(state='disabled')
        try:
            self.sim.stop()
            self.sim = None
        except:
            pass
 
    def _btn_simulate_callback(self):
        self.btn_pause.config(state='normal')
        self.btn_simulate.config(state='disabled')
        self.label_sim_total_time.config(foreground='green')
        self.sim.start()
        self.update_frontend_running = True
        self.mainframe.after(0, self.update_frontend)
        

    def _btn_pause_callback(self):
        self.btn_simulate.config(state='normal')
        self.btn_pause.config(state='disabled')
        self.label_sim_total_time.config(foreground='red')
        self.sim.pause()

    def _btn_center_callback(self):
        self.renderer.center_plot_data()
        #if self.renderer2D is not None:
            #self.renderer2D.init_plots()
    
    def _btn_reset_view_callback(self):
        self.renderer.reset_view()
        
    def btn_step_callback(self):
        self.sim.step()

    def _btn_rendering_callback(self):
        if self.render_env:
            self.set_rendering('off')
        else:
            self.set_rendering('on')


    def btn_2D_view_callback(self):
        # Initialize 2D view windows
        if self.window_2d is None:
            self.window_2d = tk.Toplevel(self.mainframe)
            self.window_2d.protocol("WM_DELETE_WINDOW", self.window_2d_closing)
            self.window_2d.title("2D Swarm viewer")
            self.window_2d.geometry('1000x800')
            self.renderer2D = Renderer2D(self.window_2d, self.swarm)
            self.window_2d.bind("<KeyPress>", self.key_press_callback)
            self.window_2d.bind("<KeyRelease>", self.key_release_callback)
            self.window_2d.drone_selection_changed = self.drone_selection_changed
        self.window_2d.deiconify()


    def btn_apply_all_callback(self):
        try:
            self.swarm.set_noise(self.listbox_noise_type.get(), self.var_noise_param_dist.get(), self.var_noise_param_heading.get(), self.var_noise_param_dir.get(), apply_all=True)
            print('Noise parameters changed for ALL: {0}'.format(self.swarm.get_noise()))
        except Exception as e:
            print("Error setting noise: {0}".format(e))

    def start_recording_callback(self):
        self._recorder.start(self.get_app_params_dict())
        self.button_stop_recording.config(state='normal')
        self.button_start_recording.config(state='disabled')

    def stop_recording_callback(self):
        if not self._is_autorun:
            self._recorder.export(DATA_OUTPUT_FOLDER + '/' + self.var_output_csv.get())
        else:
            self._recorder.stop()
        self.button_stop_recording.config(state='disabled')
        self.button_start_recording.config(state='normal')

    def _btn_autorun_callback(self):
        filepath = os.path.join('./config', self.var_autorun_filename.get())
        self.autorun = AutorunSim(self, filepath)
        self._is_autorun = True
        self.autorun.run_all()

    def _btn_control_scheme_params_callback(self):
        ControlSchemePanel(self, self.var_swarming_algorithm.get())

    def _add_obstacle_callback(self):
        # Create a new obstacle with current parameters
        try:
            x = float(self.spinner_obstacle_x.get())
            y = float(self.spinner_obstacle_y.get())
            z = float(self.spinner_obstacle_z.get())
            height = float(self.spinner_obstacle_height.get())
            radius = float(self.spinner_obstacle_radius.get())
        except ValueError as e:
            return
        self.env.add_obstacle('cylinder', center=[x, y, z], height=height, radius=radius, selected=True)
        self.treeview_obstacles.insert('', 'end', text=len(self.env.obstacles)-1, values=('cylinder', [x, y, z], height, radius))
        # select the new obstacle
        self.treeview_obstacles.selection_set(self.treeview_obstacles.get_children()[-1])
        self.btn_remove_obstacle.config(state='normal')

    def _remove_obstacle_callback(self):
        if len(self.treeview_obstacles.selection()) == 0:
            self.btn_remove_obstacle.config(state='disabled')
            return
        idx_to_remove = []
        for item in self.treeview_obstacles.selection():
            idx_to_remove.append(self.treeview_obstacles.item(item, 'text'))
        self.env.remove_obstacle(idx_to_remove)
        self._generate_obstacles_list()

    def _save_env_callback(self):
        # For now only export obstacles definition
        self.app_config['obstacles'] = [obs.to_dict() for obs in self.env.obstacles]
        self.export_app_config()


    def _btn_concave_hull_callback(self):
        if self.swarm is None:
            return
        self.swarm.concave_hull_enabled = not self.swarm.concave_hull_enabled
        if self.swarm.concave_hull_enabled:
            alpha = self.var_alpha.get()
            self.swarm.set_alpha(alpha) 
            self.btn_toggle_hull.config(text='Disable Concave Hull')
        else:
            self.btn_toggle_hull.config(text='Enable Concave Hull')


    #==================================#
    #          KEYBOARD CALLBACKS      #
    #==================================#
    
    def key_press_callback(self, event):
        if self.swarm is None:
            return
        if self.var_trajectory_mode.get().upper() != 'KEYBOARD':
            return
        target_vel = self.swarm.get_cmd_velocity()
        vref = self.app_config['simulation'].get("vref", 1.0)
        match event.char:
            case 'q':
                self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,self.cmd_yaw]))
            case 'e':
                self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,-self.cmd_yaw]))
            case 'w':
                target_vel[0] = vref
            case 's':
                target_vel[0] = -vref
            case 'a':
                target_vel[1] = vref
            case 'd':
                target_vel[1] = -vref
            case 'Q':
                target_vel[2] = self.cmd_vel
            case 'E':
                target_vel[2] = -self.cmd_vel
        
        self.swarm.set_cmd_velocity(target_vel)   


    def key_release_callback(self, event):
        if self.swarm is None:
            return
        if self.var_trajectory_mode.get().upper() != 'KEYBOARD':
            return
        target_vel = self.swarm.get_cmd_velocity()
        match event.char:
            case 'q' | 'e':
                self.swarm.set_cmd_ang_rates(np.array([0.0,0.0,0.0]))
            case 'w' | 's':
                target_vel[0] = 0
            case 'a' | 'd':
                target_vel[1] = 0
            case 'Q' | 'E':
                target_vel[2] = 0
        self.swarm.set_cmd_velocity(target_vel)  

    #==================================#
    #          OTHER EVENTS            #
    #==================================#  

    def app_closing(self):
        self.update_frontend_running = False
        try:
            if self.render_env:
                self.renderer.stop()
            if self.autorun is not None:
                self.autorun.stop()
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

    def update_frontend(self):
        try:
            self.update_sim_data(self.sim.get_total_time())
        except:
            pass
        if self.update_frontend_running:
            self.mainframe.after(int(FRONTEND_UPDATE_INTERVAL*1000), self.update_frontend)

    def get_app_params_dict(self):
        return {
            "drone_count": self.var_drone_count.get(),
            "swarming_algorithm": {
                "name": self.var_swarming_algorithm.get(),
                "params": self.app_config['swarming_algorithm']['params']
            },
            "target": self.textbox_target.get(),
            "neighbors": {
                "sampling": self.var_neighbors_sampling.get(),
                "count": self.var_neighbors_count.get(),
                "sensing_range": self.var_neighbors_sensing_range.get(),
                "r_agent": self.var_neighbors_r_agent.get(),
                "metric": self.listbox_neighbors_algo.get(),
                "computation": self.listbox_neighbors_select.get()
            },
            "noise": {
                "type": self.listbox_noise_type.get(),
                "param_dist": self.var_noise_param_dist.get(),
                "param_dir" : self.var_noise_param_dir.get(),
                "param_heading": self.var_noise_param_heading.get()
            },
            "viewing_metric": {
                "algorithm": self.var_viewing_metric_algorithm.get(),
                "outer_points": self.var_viewing_metric_outer_points.get(),
                "faces": self.var_viewing_metric_faces.get(),
                "dim": self.var_viewing_metric_dim.get()
            },
            "simulation": {
                "render": self.render_env,
                "pd_controller": self.app_config['simulation']['pd_controller'],
                "trajectory_mode": self.var_trajectory_mode.get(),
                "trajectory_type": self.listbox_trajectories.get()
            }
        }
    
    def set_app_params_dict_values(self, params=None):
        if params is not None:
            self.app_config.update(params)
        self.var_drone_count.set(self.app_config.get('drone_count', 10))
        self.textbox_target.delete(0, tk.END)
        self.textbox_target.insert(0, self.app_config.get('target', ''))
        self.var_neighbors_sampling.set(self.app_config['neighbors'].get('sampling', 1))
        self.var_neighbors_count.set(self.app_config['neighbors'].get('count', 1))
        self.var_neighbors_sensing_range.set(self.app_config['neighbors'].get('sensing_range', 1.0))
        self.var_neighbors_r_agent.set(self.app_config['neighbors'].get('r_agent', 1.0))
        self.listbox_neighbors_algo.set(self.app_config['neighbors'].get('metric', 'Eucledian'))
        self.listbox_neighbors_select.set(self.app_config['neighbors'].get('computation', 'None'))
        self.listbox_noise_type.set(self.app_config['noise'].get('type', 'None'))
        self.var_noise_param_dist.set(self.app_config['noise'].get('param_dist', 0.0))
        self.var_noise_param_dir.set(self.app_config['noise'].get('param_dir', 0.0))
        self.var_noise_param_heading.set(self.app_config['noise'].get('param_heading', 0.0))
        self.var_swarming_algorithm.set(self.app_config['swarming_algorithm'].get('name', 'olfati-saber'))
        if self.swarm is not None:
            self.var_viewing_metric_algorithm.set(self.app_config['viewing_metric'].get('algorithm', 'Outer'))
            self.var_viewing_metric_outer_points.set(self.app_config['viewing_metric'].get('outer_points', 10))
            self.var_viewing_metric_faces.set(self.app_config['viewing_metric'].get('faces', 10))
            self.var_viewing_metric_dim.set(self.app_config['viewing_metric'].get('dim', 3))
        
    
    def set_var_value(self, param, value):
        self.app_config.update({param: value})
        param = "var_" + param.lower()
        try:
            getattr(self, param).set(value)
        except:
            print(f'Error setting value: {param}={value}')

    def get_var_ref(self, param):
        param = "var_" + param.lower()
        try:
            return getattr(self, param)
        except:
            print(f'Error getting value: {param}')

    def set_rendering(self, val):
        if val.upper() == "OFF":
            self.render_env = False
            self.btn_rendering.config(text="Rendering: OFF")
            if self.renderer is not None:
                self.renderer.disable_rendering()
            self.renderer = None
            self.label_no_renderering.grid(column=0,row=0,sticky='NESW')
            for child in self.panel_3D_quick_views.winfo_children():
                child.config(state='disabled')
        elif val.upper() == "ON":
            self.render_env = True
            self.btn_rendering.config(text="Rendering: ON")
            self.label_no_renderering.grid_forget()
            self.renderer = Renderer(self.panel_view, self.swarm, callbacks=self.RENDERER_CALLBACKS, env=self.env)
            for child in self.panel_3D_quick_views.winfo_children():
                child.config(state='normal')

    def save_recording(self):
        self._recorder.export(DATA_OUTPUT_FOLDER + '/' + self.var_output_csv.get())
        self._recorder.clear()

    def export_app_config(self):
        self.app_config.update(self.get_app_params_dict())
        with open(os.path.join('./pyswarm_sim/config', CONFIG_FILENAME), 'w') as f:
            json.dump(self.app_config, f, indent=4)
    

if __name__ == "__main__":
    root = tk.Tk()
    #if len(sys.argv) > 1:
    #    # Set width and height
    #    root.geometry('{0}x{1}+0+0'.format(sys.argv[1], sys.argv[2]))
    root.title("Swarm boundaries simulation")
    root.geometry('{0}x{1}+0+0'.format(w, h))
    app = myApp(root)

    root.mainloop()