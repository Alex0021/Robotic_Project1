####################################################
# THIS FILE CONTAINS THE GUI OF THE SIMULATOR
# FOR MY SEMESTER PROJECT
####################################################
import sys, os
sys.path.insert(0, './src')

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from src.swarm import *
from src.renderer import *
from src.simulator import Simulator
from src.recorder import SwarmRecorder
import json
import time
from src.tester import AutorunSim

w,h = (1600,800)
CONFIG_FILENAME = 'app_config.json'     # Default config filename
AUTORUN_FILENAME = 'sim_autorun.json'   # Default auorun filename
DATA_OUTPUT_FOLDER = 'sim_results'      # Output folder for .npy data files after recording
FRONTEND_UPDATE_INTERVAL = 0.1          # To querry updates for the algo panel sidebar (viewing directions, coverage %)


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
        self.mainframe.drone_selection_changed = self.drone_selection_changed

        self.window_2d = None
        self.renderer2D = None
        self.swarm = None
        self.autorun = None

        #==================================#
        # JSON CONFIG FILE INITIALIZATION  #
        #==================================#
        try:
            with open(os.path.join('./config', CONFIG_FILENAME)) as f:
                self.app_config = json.load(f)
        except FileNotFoundError:
            self.app_config = {}
        # Initialize app variables with json values or defaults
        self.var_drone_count = tk.IntVar(value=self.app_config.get('drone_count', 10))
        self.var_neighbors_metric = tk.StringVar(value=self.app_config['neighbors'].get('metric', 'Topological'))
        self.var_neighbors_count = tk.IntVar(value=self.app_config['neighbors'].get('count', 1))
        self.var_neighbors_sensing_range = tk.DoubleVar(value=self.app_config['neighbors'].get('sensing_range', 1.0))
        self.var_neighbors_r_agent = tk.DoubleVar(value=self.app_config['neighbors'].get('r_agent', 0.01))
        self.var_neighbors_sampling = tk.IntVar(value=self.app_config['neighbors'].get('sampling', 1))
        self.var_swarm_spread = tk.DoubleVar(value=self.app_config.get('swarm_spread', 1.0))
        self.var_noise_param_dist = tk.DoubleVar(value=self.app_config['noise'].get('param_dist', 0.05))
        self.var_noise_type = tk.StringVar(value=self.app_config['noise'].get('type', 'None'))
        self.var_noise_param_dir = tk.DoubleVar(value=self.app_config['noise'].get('param_dir', 0.02))
        self.var_delta = tk.DoubleVar(value=self.app_config.get('delta', 0.0))
        self.var_r_coh = tk.DoubleVar(value=self.app_config.get('r_coh', 0.0))
        self.var_vref = tk.DoubleVar(value=self.app_config.get('vref', 0.0))
        self.var_a = tk.DoubleVar(value=self.app_config.get('a', 0.0))
        self.var_b = tk.DoubleVar(value=self.app_config.get('b', 0.0))
        self.var_z_offset = tk.DoubleVar(value=self.app_config.get('z_offset', 10.0))
        self.spawn_area = self.app_config.get('spawn_area', [0,0,0,1])
        self.cmd_yaw = self.app_config.get('cmd_yaw', 0.5)
        self.cmd_vel = self.app_config.get('cmd_vel', 0.5)
        self.var_viewing_metric_outter_points = tk.IntVar(value=self.app_config['viewing_metric'].get('outter_points', 2))
        self.var_output_csv = tk.StringVar(value=self.app_config.get('output_data', 'output'))
        self.render_env = self.app_config['simulation'].get('render', True)
        self.trajectory_mode_enabled = self.app_config['simulation'].get('trajectory_mode', False)
        self.var_viewing_metric_dim = tk.IntVar(value=self.app_config['viewing_metric'].get('dim', 2))
        self.var_viewing_metric_algorithm = tk.StringVar(value=self.app_config['viewing_metric'].get('algorithm', 'None'))
        self.var_viewing_metric_faces = tk.StringVar(value=self.app_config['viewing_metric'].get('faces', 'adjacent'))
        self.var_autorun_filename = tk.StringVar(value=AUTORUN_FILENAME)
        self.var_sim_dt = tk.DoubleVar(value=self.app_config['simulation'].get('dt', 0.01))

        #==================================#
        #   APP VARIABLES TRACKING         #
        #==================================#
        self.var_drone_count.trace_add('write', self._update_swarm_count)
        self.var_neighbors_metric.trace_add('write', self._update_neighbors_panel_components)
        self.var_neighbors_count.trace_add('write', self._set_neighbors_algo_params)
        self.var_swarm_spread.trace_add('write', self._set_swarm_algo_params)
        self.var_delta.trace_add('write', self._set_swarm_algo_params)
        self.var_r_coh.trace_add('write', self._set_swarm_algo_params)
        self.var_vref.trace_add('write', self._set_swarm_algo_params)
        self.var_a.trace_add('write', self._set_swarm_algo_params)
        self.var_b.trace_add('write', self._set_swarm_algo_params)
        self.var_noise_type.trace_add('write', self.noise_changed_callback)
        self.var_noise_param_dist.trace_add('write', self.noise_changed_callback)
        self.var_noise_param_dir.trace_add('write', self.noise_changed_callback)
        self.var_neighbors_sampling.trace_add('write', self._set_neighbors_algo_params)
        self.var_neighbors_sensing_range.trace_add('write', self._set_neighbors_algo_params)
        self.var_neighbors_r_agent.trace_add('write', self._set_neighbors_algo_params)
        self.var_viewing_metric_outter_points.trace_add('write', self.viewing_metric_changed_callback)
        self.var_viewing_metric_algorithm.trace_add('write', self.viewing_metric_changed_callback)
        self.var_viewing_metric_faces.trace_add('write', self.viewing_metric_changed_callback)
        self.var_sim_dt.trace_add('write', self._update_sim_dt)
        #==================================#
        #       APP INITIALIZATION         #
        #==================================#
        self.init_main_panels()
        self.init_algo_components()
        self.init_sidebar_components()
        self.noise_changed_callback()
        # Start 3D plot renderer
        if self.render_env:
            self.renderer = Renderer(self.panel_view, None)
        # Check for target
        self.textbox_target.delete(0, tk.END)
        self.textbox_target.insert(0, self.app_config.get('target', ''))
        self.swarm_2d = self.app_config.get('swarm_2d', True)
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

        # Panel global params
        self.panel_sidebar = tk.Frame(self.mainframe, bg='lightgray')
        self.panel_sidebar.grid_columnconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure(0,weight=1)
        self.panel_sidebar.grid_rowconfigure(1,weight=6)
        self.panel_sidebar.grid_rowconfigure(2,weight=4)
        self.tabbed_pane.add(self.panel_sidebar, text='Swarm config')

        # Panel alog params
        self.panel_algo = tk.Frame(self.mainframe, bg='lightgray')
        self.panel_algo.grid_columnconfigure((0,1,2),weight=1)
        self.panel_algo.grid_rowconfigure(list(range(10)),weight=1)
        self.panel_algo.grid_rowconfigure(10,weight=2)
        self.tabbed_pane.add(self.panel_algo, text='Algo params')

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

    def init_algo_components(self):
        self.label_algo = ttk.Label(self.panel_algo, anchor='w', text="Algorithm choice: ", justify='left', font=font.Font(size=14))
        self.label_algo.grid(column=0,row=0, sticky='NEWS')
        self.listbox_viewing_algo = ttk.Combobox(self.panel_algo, values=["None", "average", "outter", "tangent_plane", "convex_hull"], state='disabled', font=font.Font(size=14))
        self.listbox_viewing_algo.set(self.var_viewing_metric_algorithm.get())
        self.listbox_viewing_algo.grid(column=1,row=0, sticky='W', padx=5)
        self.listbox_viewing_algo.bind("<<ComboboxSelected>>", lambda _: self.var_viewing_metric_algorithm.set(self.listbox_viewing_algo.get()))
        self.label_algo_param = ttk.Label(self.panel_algo, anchor='w', text="Algo params:", justify='left', font=font.Font(size=14))
        self.label_algo_param.grid(column=0,row=1, sticky='NEWS')

        # Algo params for outter metric
        self.panel_algo_params_outter = tk.Frame(self.panel_algo, borderwidth=2, relief='ridge')
        self.panel_algo_params_outter.grid_columnconfigure((0,1), weight=1)
        self.panel_algo_params_outter.grid_rowconfigure(0, weight=1)
        self.label_algo_param_outter_points = ttk.Label(self.panel_algo_params_outter, anchor='w', text="# points: ", font=font.Font(size=14))
        self.label_algo_param_outter_points.grid(column=0,row=0, sticky='NEWS')
        self.spinner_algo_param_outter_points = ttk.Spinbox(self.panel_algo_params_outter, increment=1, from_=2, to=10, textvariable=self.var_viewing_metric_outter_points, font=font.Font(size=12), width=20)
        self.spinner_algo_param_outter_points.grid(column=1,row=0, sticky='W', padx=5)

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
        self.label_drone_nb.grid(column=0,row=0, sticky='NEWS')
        self.spinner_drone_nb = ttk.Spinbox(self.panel_params, increment=1,from_=1, to=50, command=lambda: self.var_drone_count.set(self.spinner_drone_nb.get()))
        self.spinner_drone_nb.bind("<Return>", lambda e: self.var_drone_count.set(self.spinner_drone_nb.get()))
        self.spinner_drone_nb.grid(row=0,column=1,sticky='w', padx=5)
        self.spinner_drone_nb.set(self.var_drone_count.get())

        # Neighbors
        self.panel_neighbors = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
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
        self.textbox_control_r_coh = ttk.Entry(self.panel_control_scheme, width=10, textvariable=self.var_r_coh)
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
        
        # Noise
        self.pnael_noise = tk.Frame(self.panel_params, borderwidth=2, relief='ridge')
        self.pnael_noise.grid(column=0, row=6, columnspan=3, rowspan=2, sticky='NWES', padx=0, pady=5)
        self.pnael_noise.grid_columnconfigure((0,1,2,3,4,5,6), weight=1)
        self.pnael_noise.grid_rowconfigure((0,1), weight=1)
        self.label_noise = ttk.Label(self.pnael_noise, anchor='w', text="Noise: ")
        self.label_noise.grid(column=0,row=0, sticky='NEWS')
        self.listbox_noise_type = ttk.Combobox(self.pnael_noise, values=["None", "Uniform", "Gaussian"])
        self.listbox_noise_type.set(self.app_config['noise'].get('type', 'None'))
        self.listbox_noise_type.grid(column=1,row=0, sticky='we', padx=5)
        self.listbox_noise_type.bind("<<ComboboxSelected>>", lambda e: self.var_noise_type.set(self.listbox_noise_type.get()))
        self.btn_apply_noise_all = ttk.Button(self.pnael_noise, text="Apply to all", command=self.btn_apply_all_callback)
        self.btn_apply_noise_all.grid(column=3,row=0, columnspan=2, sticky='NEWS', pady=10)
        self.label_noise_pos = ttk.Label(self.pnael_noise, anchor='e', text="Dist:", justify='right')
        self.label_noise_pos.grid(column=0,row=1, sticky='NEWS')
        self.spinner_noise_pos = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_param_dist)
        self.spinner_noise_pos.grid(column=1, row=1, sticky='W', padx=5)
        self.label_noise_heading = ttk.Label(self.pnael_noise, anchor='w', text="Dir (sensing cone): ", justify='right')
        self.label_noise_heading.grid(column=2,row=1, sticky='NEWS')
        self.spinner_noise_heading = ttk.Spinbox(self.pnael_noise, increment=0.01, from_=0, to=1, textvariable=self.var_noise_param_dir)
        self.spinner_noise_heading.grid(column=3, row=1, sticky='W', padx=5)
        
        # Target
        self.panel_target = tk.Frame(self.panel_params)
        self.panel_target.grid(column=0, row=8, columnspan=3, rowspan=1, sticky='NWES', padx=0, pady=0)
        self.panel_target.grid_columnconfigure((0,1,2,4), weight=1)
        self.panel_target.grid_rowconfigure(0, weight=1)
        self.label_target = ttk.Label(self.panel_target, anchor='w', text="Target: ")
        self.label_target.grid(column=0,row=0, sticky='NEWS')
        self.textbox_target = ttk.Entry(self.panel_target, text='', font=font.Font(size=10))
        self.textbox_target.bind("<Return>", self._set_swarm_algo_params)
        self.textbox_target.grid(column=1, row=0, sticky='WE', padx=5)
        self.label_target_format = ttk.Label(self.panel_target, anchor='w', text="#.#;#.#;#.# or empty", justify='left', font=font.Font(size=10))
        self.label_target_format.grid(column=2,row=0, sticky='NEWS')
        btn_text = "Single mode" if self.trajectory_mode_enabled else "Trajectory mode"
        self.btn_trajectory_mode = ttk.Button(self.panel_target, text=btn_text, command=self.btn_trajectory_mode_callback, state='disabled')
        self.btn_trajectory_mode.grid(column=4,row=0, sticky='NEWS', padx=5)

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
        self.btn_center.grid(column=2, row=2, sticky='EW', padx=10, pady=5)
        self.btn_reset_view = ttk.Button(self.panel_sim, text="Reset view", command=self._btn_reset_view_callback, state='disabled')
        self.btn_reset_view.grid(column=1, row=2, sticky='EW', padx=10, pady=5)
        text_btn_rendering = "Rendering: ON" if self.render_env else "Rendering: OFF"
        self.btn_rendering = ttk.Button(self.panel_sim, text=text_btn_rendering, command=self._btn_rendering_callback)
        self.btn_rendering.grid(column=2, row=1, sticky='EW', padx=10, pady=5)

        # Simulation timestep
        self.panel_sim_timestep = tk.Frame(self.panel_sim)
        self.panel_sim_timestep.grid(column=0, row=2,sticky='NWES')
        self.panel_sim_timestep.grid_columnconfigure(0, weight=1)
        self.panel_sim_timestep.grid_columnconfigure(1, weight=5)
        self.panel_sim_timestep.grid_rowconfigure(0, weight=1)
        self.label_sim_dt = ttk.Label(self.panel_sim_timestep, text="dt: ", justify='right')
        self.label_sim_dt.grid(column=0, row=0, sticky='E')
        self.spinner_sim_dt = ttk.Spinbox(self.panel_sim_timestep, increment=0.001, from_=0.001, to=1, textvariable=self.var_sim_dt, width=10)
        self.spinner_sim_dt.grid(column=1, row=0, sticky='W', padx=5)
        self.label_sim_total_time = ttk.Label(self.panel_sim, text='Simulation time: 0.000 s', justify='left', font=font.Font(size=10))
        self.label_sim_total_time.grid(column=0, row=3, columnspan=3, sticky='NEWS', padx=5)

        # Neihbors display
        self.panel_sim_neighbors = tk.Frame(self.panel_sim)
        self.panel_sim_neighbors.grid_columnconfigure((0,1), weight=1)
        self.panel_sim_neighbors.grid_columnconfigure(2, weight=2)
        self.panel_sim_neighbors.grid_rowconfigure(0, weight=1)
        self.panel_sim_neighbors.grid(row=3, column=1, columnspan=2, sticky='NWES')
        self.label_neighbor_sampling = ttk.Label(self.panel_sim_neighbors, text="Neighbor sampling: ", justify='right')
        self.label_neighbor_sampling.grid(column=0, row=0, sticky='E')
        self.spinner_neighbor_sampling = ttk.Spinbox(self.panel_sim_neighbors, increment=1, from_=1, to=100, textvariable=self.var_neighbors_sampling)
        self.spinner_neighbor_sampling.grid(column=1, row=0, sticky='W', padx=5)
        self.listbox_neighbors_select = ttk.Combobox(self.panel_sim_neighbors, values=["None", "Selected", "All"])
        self.listbox_neighbors_select.set(self.app_config['neighbors'].get('computation', 'None'))
        self.listbox_neighbors_select.bind("<<ComboboxSelected>>", lambda e: self._set_neighbors_algo_params())
        self.listbox_neighbors_select.grid(column=2, row=0, sticky='WE', padx=5)


    #==================================#
    #   SIM COMPONENTS CALLBACK        #
    #==================================# 
        
    def _initialize_simulation(self, verbose=True):
        # Retrieve all necessary parameters from app widgets
        nb_drones = self.var_drone_count.get()
        target_text = self.textbox_target.get()
        if target_text == '':
            target = None
        else:
            target_text = target_text.strip('()')
            target_numbers = target_text.split(';')
            target = np.asarray(target_numbers, dtype=float)

        self.swarm = Swarm(count=nb_drones, area=self.spawn_area, migration_point=target, in_2d=self.swarm_2d)
        dt = float(self.var_sim_dt.get())
        self._recorder._swarm = self.swarm
        self.sim = Simulator(dt, self.swarm, self._recorder)
        #self.swarm.initialize_random_vel([0.1, 0.5, -0.3, 0.3, 0, 0.2])
        self._set_neighbors_algo_params()
        self._set_swarm_algo_params()
        self.swarm.initialize_members()
        self.swarm.use_pd_smoothing(self.app_config['simulation'].get('use_pd_smoothing', False))
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
        self.btn_trajectory_mode.config(state='normal')
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
        self.btn_reset_view.config(state='normal')
        if self.swarm.is_2D:
            self.var_viewing_metric_dim.set(2)
            self.radio_3D_viewing.config(state='disabled')
        if self.trajectory_mode_enabled:
            self.swarm.set_migration_mode('trajectory')
        self.viewing_metric_changed_callback(None)
        

    def _set_swarm_algo_params(self, *args):
        if self.swarm is None:
            return

        try:
            self.swarm.algo_params.update( {
                'delta': self.var_delta.get(),
                'd_ref': self.var_swarm_spread.get(),
                'a': self.var_a.get(),
                'b': self.var_b.get(),
                'r0_coh': self.var_r_coh.get(),
                'v_ref_target': self.var_vref.get()
            })
        except ValueError as e:
            pass
        except Exception as e:
            print("Error setting swarm algo params: {0}".format(e))
        try:
            target = self.textbox_target.get().split(';')
            target = np.asarray(target, dtype=float)
            self.swarm.migration_point = target
        except ValueError as e:
            pass

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
        if self.var_noise_type.get() == 'None':
            self.spinner_noise_pos.config(state='disabled')
            self.spinner_noise_heading.config(state='disabled')
        else:
            self.spinner_noise_pos.config(state='normal')
            self.spinner_noise_heading.config(state='normal')
        if self.swarm is None:
            return
        try:
            self.swarm.set_noise(self.listbox_noise_type.get(), self.var_noise_param_dist.get(), self.var_noise_param_dir.get())
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
        self.var_noise_param_dist.set(noise.get('param_dist', self.app_config['noise'].get('param_dist', 0.05)))
        self.var_noise_param_dir.set(noise.get('param_dir', self.app_config['noise'].get('param_dir', 0.05)))

    def viewing_metric_changed_callback(self, *args):
        # Viewing algorithm
        algo = self.var_viewing_metric_algorithm.get()
        self.listbox_viewing_algo.set(algo)
        self.listbox_convex_hull_faces.set(self.var_viewing_metric_faces.get())
        if algo.upper() == 'OUTTER':
            self.panel_algo_params_outter.grid(column=1, row=1, sticky='NWES', padx=5, pady=0)
        else:
            self.panel_algo_params_outter.grid_forget()
        if algo.upper() == 'CONVEX_HULL':
            self.panel_algo_params_convex_hull.grid(column=1, row=1, sticky='NWES', padx=5, pady=0)
        else:
            self.panel_algo_params_convex_hull.grid_forget()
        # Other params
        dim = self.var_viewing_metric_dim.get()
        params = {'nb_points': self.var_viewing_metric_outter_points.get(), 
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
        self.btn_trajectory_mode.config(state='disabled')
        self.radio_2D_viewing.config(state='disabled')
        self.radio_3D_viewing.config(state='disabled')
        self.button_stop_recording.config(state='disabled')
        self.button_start_recording.config(state='disabled')
        self.btn_reset_view.config(state='disabled')
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
            self.swarm.set_noise(self.listbox_noise_type.get(), self.var_noise_param_dist.get(), self.var_noise_orient.get(), self.var_noise_param_dir.get(), apply_all=True)
            print('Noise parameters changed for ALL: {0}'.format(self.swarm.get_noise()))
        except Exception as e:
            print("Error setting noise: {0}".format(e))

    def btn_trajectory_mode_callback(self):
        if self.trajectory_mode_enabled:
            self.set_trajectory_mode('off')
        else:
            self.set_trajectory_mode('on')

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

    #==================================#
    #          KEYBOARD CALLBACKS      #
    #==================================#
    
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
            case 'Q':
                target_vel[2] = self.cmd_vel
            case 'E':
                target_vel[2] = -self.cmd_vel
        self.swarm.set_cmd_velocity(target_vel)   


    def key_release_callback(self, event):
        if self.swarm is None:
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
            "swarm_spread": self.var_swarm_spread.get(),
            "delta": self.var_delta.get(),
            "r_coh": self.var_r_coh.get(),
            "vref": self.var_vref.get(),
            "a": self.var_a.get(),
            "b": self.var_b.get(),
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
                "param_heading": 0.0
            },
            "viewing_metric": {
                "algorithm": self.var_viewing_metric_algorithm.get(),
                "outter_points": self.var_viewing_metric_outter_points.get(),
                "faces": self.var_viewing_metric_faces.get(),
                "dim": self.var_viewing_metric_dim.get()
            }
        }
    
    def set_var_value(self, param, value):
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
        elif val.upper() == "ON":
            self.render_env = True
            self.btn_rendering.config(text="Rendering: ON")
            self.label_no_renderering.grid_forget()
            self.renderer = self.renderer = Renderer(self.panel_view, self.swarm)

    def set_trajectory_mode(self, val: str):
        if val.upper() == "OFF":
            self.trajectory_mode_enabled = False
            self.btn_trajectory_mode.config(text='Trajectory mode')
            self.textbox_target.config(state='normal')
            self.swarm.set_migration_mode('single')
        elif val.upper() == "ON":
            self.trajectory_mode_enabled = True
            self.btn_trajectory_mode.config(text='Single mode')
            self.textbox_target.config(state='disabled')
            self.swarm.set_migration_mode('trajectory')

    def save_recording(self):
        self._recorder.export(DATA_OUTPUT_FOLDER + '/' + self.var_output_csv.get())
        self._recorder.clear()
    

if __name__ == "__main__":
    root = tk.Tk()
    #if len(sys.argv) > 1:
    #    # Set width and height
    #    root.geometry('{0}x{1}+0+0'.format(sys.argv[1], sys.argv[2]))
    root.title("Swarm boundaries simulation")
    root.geometry('{0}x{1}+0+0'.format(w, h))
    app = myApp(root)

    root.mainloop()