import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import pyswarm_sim.src.olfati_saber as olsaber
import pyswarm_sim.src.reynolds as renoylds
import json

#==============================================================================#
#   CONSTANTS                                                                  #
#==============================================================================#
D_REF_LIMITS = (0, 10)
R_COH_LIMITS = (0, 100)

class ControlSchemePanel(tk.Toplevel):
    def __init__(self, parent, current_control_scheme, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.current_app_config = parent.get_app_params_dict()
        # keep a copy of the old config if cancelled
        self.old_app_config = self.current_app_config.copy() 
        # Keep only parameters dictionnary for ease of use
        self.current_app_config = self.current_app_config['swarming_algorithm'].get('params', dict())

        # Other variables
        self.plot_canvas = None
        self.current_tab = 'Agents'

        # Tk variables
        # Olfati-Saber
        self.a_param = tk.StringVar()
        self.b_param = tk.StringVar()
        self.c_param = tk.StringVar()
        self.c_param_auto = tk.BooleanVar()
        self.delta_param = tk.StringVar()
        self.r_coh_param = tk.DoubleVar()
        self.d_ref_param = tk.DoubleVar()
        self.lambda_param = tk.DoubleVar()
        self.c_pm_param = tk.DoubleVar()
        self.c_vm_param = tk.DoubleVar()
        self._affect_values_to_vars()
        # Reynolds
        self.cohesion_param = tk.DoubleVar()
        self.cohesion_param.set(self.current_app_config.get("c_coh", 1.0))
        self.separation_param = tk.DoubleVar()
        self.separation_param.set(self.current_app_config.get("c_sep", 2.0))
        self.alignment_param = tk.DoubleVar()
        self.alignment_param.set(self.current_app_config.get("c_align", 3.0))
        self.migration_param = tk.DoubleVar()
        self.migration_param.set(self.current_app_config.get("c_mig", 4.0))
        self.obstacle_param = tk.DoubleVar()
        self.obstacle_param.set(self.current_app_config.get("c_obs", 5.0))

        # Variables binding
        self.c_param_auto.trace_add("write", self.calculate_c_param)
        self.a_param.trace_add("write", self.calculate_c_param)
        self.b_param.trace_add("write", self.calculate_c_param)
        self.d_ref_param.trace_add("write", self.on_d_ref_slider_change)
        self.r_coh_param.trace_add("write", self.generate_plot)
        self.delta_param.trace_add("write", self.generate_plot)
        self.cohesion_param.trace_add("write", self.generate_plot)
        self.separation_param.trace_add("write", self.generate_plot)
        self.alignment_param.trace_add("write", self.generate_plot)
        self.obstacle_param.trace_add("write", self.generate_plot)
        self.control_scheme = current_control_scheme

        self.init_common_components()
        if current_control_scheme.upper() == "OLFATI-SABER":
            self.calculate_c_param()

        # Behavior
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)


    def init_common_components(self):
        self.title("Control Scheme")
        self.geometry("700x450")

        # Set geometry manager
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set panel for algorithm selection
        self.algorithm_panel = tk.Frame(self)
        self.algorithm_panel.grid_columnconfigure(1, weight=1)
        self.algorithm_panel.grid_rowconfigure(2, weight=1)
        self.algorithm_panel.grid(row=0, column=0, sticky="nsew")

        # Add basics to algorithm panel
        self.control_scheme_label = tk.Label(self.algorithm_panel, text="Control Scheme: ")
        self.control_scheme_label.grid(row=0, column=0, sticky="w", padx=5, pady=10)
        self.control_scheme_cbbox = ttk.Combobox(self.algorithm_panel, values=["Olfati-Saber", "Reynolds"])
        self.control_scheme_cbbox.grid(row=0, column=1, sticky="ew", padx=5, pady=10)
        self.control_scheme_cbbox.set(self.control_scheme)
        self.control_scheme_cbbox.bind("<<ComboboxSelected>>", lambda e: self.update_control_scheme(self.control_scheme_cbbox.get()))

        self.parameters_label = tk.Label(self.algorithm_panel, text="Parameters: ")
        self.parameters_label.grid(row=1, column=0, sticky="w", padx=5)
        self.parameters_panel = self.init_parameters_panel()
        self.parameters_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Save & cancel buttons
        self.save_btn = ttk.Button(self.algorithm_panel, text="Apply", command=self.on_apply)
        self.save_btn.grid(row=3, column=0, sticky="ew", padx=5, pady=10)
        self.cancel_btn = ttk.Button(self.algorithm_panel, text="Cancel", command=self.on_cancel)
        self.cancel_btn.grid(row=3, column=1, sticky="ew", padx=5, pady=10)

        # Separator
        self.separator = ttk.Separator(self, orient="vertical")
        self.separator.grid(row=0, column=1, sticky="ns")

        # Plot view panel
        self.init_plot_view_panel()


    def init_parameters_panel(self) -> tk.Frame:
        parameters_panel = tk.Frame(self.algorithm_panel)
        if self.control_scheme.upper() == "OLFATI-SABER":
            # Create a tabbed panel for swarm and obstacle parameters
            tabbed_panel_params = ttk.Notebook(parameters_panel)
            tabbed_panel_params.grid(row=0, column=0, sticky="nsew")
            self.tab1_panel = tk.Frame(tabbed_panel_params, border=2, relief="groove")
            self._generate_panel_params_os(self.tab1_panel)
            tabbed_panel_params.add(self.tab1_panel, text="Agents")
            self.tab2_panel = tk.Frame(tabbed_panel_params, border=2, relief="groove")
            # self._generate_panel_params_os(self.tab2_panel)
            tabbed_panel_params.add(self.tab2_panel, text="Obstacles")
            tabbed_panel_params.bind("<<NotebookTabChanged>>", self.notebook_tab_changed)

        elif self.control_scheme.upper() == "REYNOLDS":
            parameters_panel.grid_columnconfigure(0, weight=0)
            parameters_panel.grid_columnconfigure(1, weight=1)
            parameters_panel.configure(border=2, relief="groove")

            # Cohesion parameter
            self.cohesion_label = tk.Label(parameters_panel, text="Cohesion: ")
            self.cohesion_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.cohesion_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.cohesion_param)
            self.cohesion_spinner.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

            # Separation parameter
            self.separation_label = tk.Label(parameters_panel, text="Separation: ")
            self.separation_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.separation_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.separation_param)
            self.separation_spinner.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

            # If migration point is not None, use migration parameter instead of alignment
            if len(self.current_app_config.get("target", "")) == 0:
                # Alignment parameter
                self.alignment_label = tk.Label(parameters_panel, text="Alignment: ")
                self.alignment_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
                self.alignment_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.alignment_param)
                self.alignment_spinner.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
            else:
                # Migration parameter
                self.migration_label = tk.Label(parameters_panel, text="Migration: ")
                self.migration_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
                self.migration_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.migration_param)
                self.migration_spinner.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

            # Obstacle parameter
            self.obstacle_label = tk.Label(parameters_panel, text="Obstacle: ")
            self.obstacle_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
            self.obstacle_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.obstacle_param)
            self.obstacle_spinner.grid(row=3, column=1, sticky="ew", padx=5, pady=5)


        return parameters_panel
    
    def _generate_panel_params_os(self, parent: tk.Frame):
        parameters_panel = parent
        # d_ref parameter
        self.d_ref_label = tk.Label(parameters_panel, text="d_ref: ")
        self.d_ref_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.d_ref_panel = tk.Frame(parameters_panel)
        self.d_ref_panel.grid_columnconfigure(0, weight=1)
        self.d_ref_panel.grid_rowconfigure(0, weight=1)
        self.d_ref_panel.grid(row=0, column=1, columnspan=1, sticky="nsew")
        self.d_ref_slider = ttk.Scale(self.d_ref_panel, from_=D_REF_LIMITS[0], to=D_REF_LIMITS[1], orient="horizontal", variable=self.d_ref_param)
        self.d_ref_slider.grid(row=0, column=0, sticky="ew", padx=5)
        self.d_ref_value_label = tk.Label(self.d_ref_panel, text=str.format("{:.2f}", self.d_ref_param.get()))
        self.d_ref_value_label.grid(row=0, column=1, sticky="ew", padx=5)
        # r_coh parameter
        self.r_coh_label = tk.Label(parameters_panel, text="r_coh: ")
        self.r_coh_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.r_coh_spinner = ttk.Spinbox(parameters_panel, from_=R_COH_LIMITS[0], to=R_COH_LIMITS[1], increment=0.1, textvariable=self.r_coh_param)
        self.r_coh_spinner.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        # delta parameter
        self.delta_label = tk.Label(parameters_panel, text="delta: ")
        self.delta_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.delta_textbox = tk.Entry(parameters_panel, textvariable=self.delta_param)
        self.delta_textbox.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        # a parameter
        self.a_label = tk.Label(parameters_panel, text="a: ")
        self.a_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.a_textbox = tk.Entry(parameters_panel, textvariable=self.a_param)
        # self.a_textbox.insert(0, self.current_app_config.get("a", 1.0))
        self.a_textbox.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        # b parameter
        self.b_label = tk.Label(parameters_panel, text="b: ")
        self.b_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.b_textbox = tk.Entry(parameters_panel, textvariable=self.b_param)
        # self.b_textbox.insert(0, self.current_app_config.get("b", 1.0))
        self.b_textbox.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        # c parameter panel
        self.c_label = tk.Label(parameters_panel, text="c: ")
        self.c_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.c_param_panel = tk.Frame(parameters_panel)
        self.c_param_panel.grid_columnconfigure(1, weight=1)
        self.c_param_panel.grid_rowconfigure(0, weight=1)
        self.c_param_panel.grid(row=5, column=1, columnspan=1, sticky="nsew", padx=5, pady=5) 
        self.c_textbox = tk.Entry(self.c_param_panel, textvariable=self.c_param)
        self.c_textbox.grid(row=0, column=0, sticky="ew", padx=5)
        self.c_auto_checkbox = tk.Checkbutton(self.c_param_panel, text="Auto", variable=self.c_param_auto)
        self.c_auto_checkbox.grid(row=0, column=1, sticky="ew", padx=5)
        
        self.lambda_label = tk.Label(parameters_panel, text="lambda: ")
        self.lambda_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.lambda_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.lambda_param)
        self.lambda_spinner.grid(row=6, column=1, sticky="w", padx=5, pady=5)

        self.c_pm_label = tk.Label(parameters_panel, text="c_pm: ")
        self.c_pm_label.grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.c_pm_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.c_pm_param)
        self.c_pm_spinner.grid(row=7, column=1, sticky="w", padx=5, pady=5)

        self.c_vm_label = tk.Label(parameters_panel, text="c_vm: ")
        self.c_vm_label.grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.c_vm_spinner = ttk.Spinbox(parameters_panel, from_=0, to=np.inf, increment=0.1, textvariable=self.c_vm_param)
        self.c_vm_spinner.grid(row=8, column=1, sticky="w", padx=5, pady=5)

    
    def init_plot_view_panel(self):
        """
        Initialize the plot view panel for the Potential/Force function
        """
        # panel
        self.plot_view_panel = tk.Frame(self)
        self.plot_view_panel.grid_columnconfigure(0, weight=1)
        self.plot_view_panel.grid_rowconfigure(1, weight=1)
        self.plot_view_panel.grid(row=0, column=2, sticky="nsew")

        # Title
        self.plot_view_title = tk.Label(self.plot_view_panel, text="Potential Function")
        self.plot_view_title.grid(row=0, column=0, sticky="ew")
        
        # Canvas
        self.plot_view_canvas = None
        self.fig_ref = plt.figure(figsize=(3, 3))
        self.fig_ref.add_subplot(111)
        self.plot_view_panel.plot_view_canvas = FigureCanvasTkAgg(self.fig_ref, master=self.plot_view_panel)
        self.plot_view_panel.plot_view_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.plot_view_panel.plot_view_toolbar = NavigationToolbar2Tk(self.plot_view_panel.plot_view_canvas, self.plot_view_panel, pack_toolbar=False)
        self.plot_view_panel.plot_view_toolbar.grid(row=2, column=0, sticky="ew")
        self.generate_plot()
    
    def generate_plot(self, *args):
        """
        Generate the plot based on the control scheme
        """
        self.fig_ref.clear()
        if self.control_scheme.upper() == "OLFATI-SABER":
            self.generate_plot_os()
        elif self.control_scheme.upper() == "REYNOLDS":
            self.generate_plot_reynolds()
        self.plot_view_panel.plot_view_canvas.draw()
        self.plot_view_panel.plot_view_toolbar.update()
    
    def generate_plot_os(self):
        """
        Generate the plot for the Olfati-Saber control scheme
        """
        ax = self.fig_ref.add_subplot(111)
        r_coh = self.r_coh_param.get()
        d_ref = self.d_ref_param.get()
        a = float(self.a_param.get().strip())
        b = float(self.b_param.get().strip())
        c = float(self.c_param.get().strip())
        delta = float(self.delta_param.get().strip())
        x = np.linspace(0, r_coh, int(r_coh/0.01))
        w, sigma = np.zeros_like(x), np.zeros_like(x)
        for i in range(len(x)):
            w[i] = olsaber.get_neighbour_weight(x[i], r_coh, delta)
            sigma[i] = olsaber.get_cohesion_intensity(x[i], d_ref, a, b, c)
        ax.plot(x, w, label="Weight", color="black", linestyle='--')
        ax.plot(x, sigma, label="Cohesion", color="red", linestyle='--')
        ax.plot(x, w*sigma, label="Potential", color="blue", linestyle='-')
        ax.legend()

    def generate_plot_reynolds(self):
        """
        Generate the plot for the Reynolds control scheme
        """
        ax = self.fig_ref.add_subplot(111)
        c_coh = self.cohesion_param.get()
        c_sep = self.separation_param.get()
        r_coh = self.r_coh_param.get()
        x = np.linspace(0.1, r_coh, int(r_coh/0.01))
        f_coh, f_sep = np.zeros_like(x), np.zeros_like(x)
        for i in range(len(x)):
            f_coh[i] = renoylds.get_cohesion_force(0, x[i], c_coh)
            f_sep[i] = renoylds.get_separation_force(0, x[i], c_sep)
        ax.plot(x, f_coh, label="Cohesion", color="red", linestyle='--')
        ax.plot(x, f_sep, label="Separation", color="blue", linestyle='--')
        ax.plot(x, abs(f_coh-f_sep), label="Potential", color="green", linestyle='-')
        ax.legend()

    def calculate_c_param(self, *args):
        """
        Calculate c parameter based on the formula if auto is checked:
        c = (b - a)/(2*sqrt(a*b))
        """
        if self.c_param_auto.get():
            old_c_val = self.c_param.get()
            try:
                # calculate c value
                a = float(self.a_param.get().strip())
                b = float(self.b_param.get().strip())
                c_val = (b - a)/(2*np.sqrt(a*b))
            except:
                c_val = old_c_val
            self.c_textbox.config(state="disabled")
            self.c_param.set(str.format("{:.4f}", c_val))

        else:
            self.c_textbox.config(state="normal")
        # Update plot
        self.generate_plot()

    def save_app_params(self):
        """
        Generate the app config dictionary and save it to the parent
        """
        if self.control_scheme == 'olfati-saber':
            if self.current_tab == 'Agents':
                str_ext = ''
            elif self.current_tab == 'Obstacles':
                str_ext = '_obs'
            self.current_app_config[f"a{str_ext}"] = float(self.a_param.get())
            self.current_app_config[f"b{str_ext}"] = float(self.b_param.get())
            self.current_app_config[f"c{str_ext}"] = float(self.c_param.get())
            self.current_app_config[f"c_auto{str_ext}"] = self.c_param_auto.get()
            self.current_app_config[f"delta{str_ext}"] = float(self.delta_param.get())
            self.current_app_config[f"r_coh{str_ext}"] = float(self.r_coh_param.get())
            self.current_app_config[f"d_ref{str_ext}"] = float(self.d_ref_param.get())
            self.current_app_config[f"lambda{str_ext}"] = float(self.lambda_param.get())
            self.current_app_config[f"c_pm{str_ext}"] = float(self.c_pm_param.get())
            self.current_app_config[f"c_vm{str_ext}"] = float(self.c_vm_param.get())
        elif self.control_scheme == 'Reynolds':
            self.current_app_config["c_coh"] = self.cohesion_param.get()
            self.current_app_config["c_sep"] = self.separation_param.get()
            self.current_app_config["c_align"] = self.alignment_param.get()
            self.current_app_config["c_mig"] = self.migration_param.get()
            self.current_app_config["c_obs"] = self.obstacle_param.get()

    def update_control_scheme(self, control_scheme: str):
        """
        Update the control scheme panel for the selected control scheme
        Args:
            control_scheme (str): the selected control scheme
        """
        self.control_scheme = control_scheme
        self.parameters_panel.destroy()
        self.parameters_panel = self.init_parameters_panel()
        self.parameters_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        if control_scheme.upper() == "OLFATI-SABER":
            self.calculate_c_param()
        self.generate_plot()

    def _affect_values_to_vars(self):
        """
        Affect the values from the app config to the Tk variables
        """
        if self.current_tab == 'Agents':
            str_ext = ''
        elif self.current_tab == 'Obstacles':
            str_ext = '_obs'
        self.a_param.set(self.current_app_config.get(f"a{str_ext}", 1.0))
        self.b_param.set(self.current_app_config.get(f"b{str_ext}", 1.0))
        self.c_param.set(self.current_app_config.get(f"c{str_ext}", 1.0))
        self.c_param_auto.set(self.current_app_config.get(f"c_auto{str_ext}", False))
        self.delta_param.set(self.current_app_config.get(f"delta{str_ext}", 1.0))
        self.r_coh_param.set(self.current_app_config.get(f"r_coh{str_ext}", 1.0))
        self.d_ref_param.set(self.current_app_config.get(f"d_ref{str_ext}", 1.0))
        self.lambda_param.set(self.current_app_config.get(f"lambda{str_ext}", 1.0))
        self.c_pm_param.set(self.current_app_config.get(f"c_pm{str_ext}", 1.0))
        self.c_vm_param.set(self.current_app_config.get(f"c_vm{str_ext}", 1.0))


    #===========================================================================
    # Event Handlers
    #===========================================================================

    def on_d_ref_slider_change(self, *args):
        """
            Event handler for d_ref slider change
        """
        self.d_ref_value_label.config(text=str.format("{:.2f}", self.d_ref_param.get()))
        self.generate_plot()

    def on_parameters_change(self, *args):
        """ 
            Event handler for parameter changes
        """
        self.generate_plot()
        self.save_app_params()

    def notebook_tab_changed(self, event):
        """
            Event handler for tab change
        """
        self.save_app_params()
        self.current_tab = event.widget.tab(event.widget.select(), "text")
        if self.current_tab == 'Agents':
            w = self.tab2_panel
            o = self.tab1_panel
        elif self.current_tab == 'Obstacles':
            w = self.tab1_panel
            o = self.tab2_panel
        for c in w.winfo_children():
            c.destroy()
        self._generate_panel_params_os(o)
        self._affect_values_to_vars()
        self.generate_plot()

    def on_apply(self):
        """
            To do when apply button is clicked
        """
        self.save_app_params()
        self.parent.set_app_params_dict_values({"swarming_algorithm": { "name": self.control_scheme, 
                                                                "params": self.current_app_config }})
        self.destroy()
        # exit(0)

    def on_cancel(self):
        """
            To do when cancel button
        """ 
        self.parent.set_app_params_dict_values(self.old_app_config)
        self.destroy()
        # exit(0)



# ------------------------------- #
# --- TESTING PURPOSES ONLY ---   #
# ------------------------------- #
def dump_config(app_config):
    with open('test_dump.json', 'w') as f:
        json.dump(app_config, f, indent=4)
if __name__ == "__main__":
    root = tk.Tk()
    # load config file
    with open("pyswarm_sim/config/app_config.json", "r") as f:
        app_config = json.load(f)
    root.get_app_params_dict = lambda: app_config
    root.set_app_params_dict_values = dump_config
    ControlSchemePanel(root, "Olfati-Saber")
    root.mainloop()
