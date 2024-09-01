import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import olfati_saber as olsaber

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

        # Other variables
        self.plot_canvas = None

        # Tk variables
        self.delta_param = tk.StringVar()
        self.a_param = tk.StringVar()
        self.b_param = tk.StringVar()
        self.c_param = tk.StringVar()
        self.r_coh_param = tk.DoubleVar()
        self.d_ref_param = tk.DoubleVar()
        self.c_param_auto = tk.BooleanVar()

        # tk variables values
        self.a_param.set(self.current_app_config.get("a", 1.0))
        self.b_param.set(self.current_app_config.get("b", 1.0))
        self.c_param.set(self.current_app_config.get("c", 1.0))
        self.delta_param.set(self.current_app_config.get("delta", 1.0))
        self.c_param_auto.set(self.current_app_config.get("c_auto", True))
        self.r_coh_param.set(self.current_app_config.get("r_coh", 20.0))
        self.d_ref_param.set(self.current_app_config.get("d_ref", np.mean(D_REF_LIMITS)))

        # Variables binding
        self.c_param_auto.trace_add("write", self.calculate_c_param)
        self.a_param.trace_add("write", self.calculate_c_param)
        self.b_param.trace_add("write", self.calculate_c_param)
        self.d_ref_param.trace_add("write", lambda *args: self.d_ref_value_label.config(text=str.format("{:.2f}", self.d_ref_param.get())))
        self.control_scheme = current_control_scheme

        self.init_common_components()
        self.calculate_c_param()


    def init_common_components(self):
        self.title("Control Scheme")
        self.geometry("600x300")

        # Set geometry manager
        self.grid_columnconfigure((0,1,2), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set panel for algorithm selection
        self.algorithm_panel = tk.Frame(self)
        self.algorithm_panel.grid_columnconfigure((0,2), weight=10)
        self.algorithm_panel.grid_columnconfigure(1, weight=1)
        self.algorithm_panel.grid_rowconfigure((0,1), weight=1)
        self.algorithm_panel.grid_rowconfigure(2, weight=10)
        self.algorithm_panel.grid(row=0, column=0, sticky="nsew")

        # Add basics to algorithm panel
        self.control_scheme_label = tk.Label(self.algorithm_panel, text="Control Scheme: ")
        self.control_scheme_label.grid(row=0, column=0, sticky="w", padx=5)
        self.control_scheme_cbbox = ttk.Combobox(self.algorithm_panel, values=["Olfati-Saber", "Reynolds"])
        self.control_scheme_cbbox.grid(row=0, column=1, sticky="ew", padx=5)
        self.control_scheme_cbbox.set(self.control_scheme)

        self.parameters_label = tk.Label(self.algorithm_panel, text="Parameters: ")
        self.parameters_label.grid(row=1, column=0, sticky="w", padx=5)
        self.parameters_panel = self.init_parameters_panel()
        self.parameters_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Separator
        self.separator = ttk.Separator(self, orient="vertical")
        self.separator.grid(row=0, column=1, sticky="ns")

        # Plot view panel
        self.plot_view_panel = self.init_plot_view_panel()


    def init_parameters_panel(self) -> tk.Frame:
        if self.control_scheme == "Olfati-Saber":
            parameters_panel = tk.Frame(self.algorithm_panel)
            parameters_panel.grid_columnconfigure(0, weight=1)
            parameters_panel.grid_columnconfigure(1, weight=2)
            parameters_panel.grid_rowconfigure((0,1,2,3,4,5), weight=1)
            parameters_panel.configure(border=2, relief="groove")
            # d_ref parameter
            self.d_ref_label = tk.Label(parameters_panel, text="d_ref: ")
            self.d_ref_label.grid(row=0, column=0, sticky="w", padx=5)
            self.d_ref_panel = tk.Frame(parameters_panel)
            self.d_ref_panel.grid_columnconfigure(0, weight=2)
            self.d_ref_panel.grid_columnconfigure(1, weight=1)
            self.d_ref_panel.grid_rowconfigure(0, weight=1)
            self.d_ref_panel.grid(row=0, column=1, columnspan=1, sticky="nsew")
            self.d_ref_slider = ttk.Scale(self.d_ref_panel, from_=D_REF_LIMITS[0], to=D_REF_LIMITS[1], orient="horizontal", variable=self.d_ref_param)
            self.d_ref_slider.grid(row=0, column=0, sticky="ew", padx=5)
            self.d_ref_value_label = tk.Label(self.d_ref_panel, text=str.format("{:.2f}", self.d_ref_param.get()))
            self.d_ref_value_label.grid(row=0, column=1, sticky="ew", padx=5)
            # r_coh parameter
            self.r_coh_label = tk.Label(parameters_panel, text="r_coh: ")
            self.r_coh_label.grid(row=1, column=0, sticky="w", padx=5)
            self.r_coh_spinner = ttk.Spinbox(parameters_panel, from_=R_COH_LIMITS[0], to=R_COH_LIMITS[1], increment=0.1)
            self.r_coh_spinner.insert(0, self.current_app_config.get("r_coh", 20.0))
            self.r_coh_spinner.grid(row=1, column=1, sticky="ew", padx=5)
            # delta parameter
            self.delta_label = tk.Label(parameters_panel, text="delta: ")
            self.delta_label.grid(row=2, column=0, sticky="w", padx=5)
            self.delta_textbox = tk.Entry(parameters_panel, textvariable=self.delta_param)
            self.delta_textbox.grid(row=2, column=1, sticky="ew", padx=5)
            # a parameter
            self.a_label = tk.Label(parameters_panel, text="a: ")
            self.a_label.grid(row=3, column=0, sticky="w", padx=5)
            self.a_textbox = tk.Entry(parameters_panel, textvariable=self.a_param)
            # self.a_textbox.insert(0, self.current_app_config.get("a", 1.0))
            self.a_textbox.grid(row=3, column=1, sticky="ew", padx=5)
            # b parameter
            self.b_label = tk.Label(parameters_panel, text="b: ")
            self.b_label.grid(row=4, column=0, sticky="w", padx=5)
            self.b_textbox = tk.Entry(parameters_panel, textvariable=self.b_param)
            # self.b_textbox.insert(0, self.current_app_config.get("b", 1.0))
            self.b_textbox.grid(row=4, column=1, sticky="ew", padx=5)
            # c parameter panel
            self.c_label = tk.Label(parameters_panel, text="c: ")
            self.c_label.grid(row=5, column=0, sticky="w", padx=5)
            self.c_param_panel = tk.Frame(parameters_panel)
            self.c_param_panel.grid_columnconfigure((0,1), weight=1)
            self.c_param_panel.grid_rowconfigure(0, weight=1)
            self.c_param_panel.grid(row=5, column=1, columnspan=1, sticky="nsew")
            self.c_textbox = tk.Entry(self.c_param_panel, textvariable=self.c_param)
            self.c_textbox.grid(row=0, column=0, sticky="ew", padx=5)
            self.c_auto_checkbox = tk.Checkbutton(self.c_param_panel, text="Auto", variable=self.c_param_auto)
            self.c_auto_checkbox.grid(row=0, column=1, sticky="ew", padx=5)

        return parameters_panel
    
    def init_plot_view_panel(self):
        # panel
        plot_view_panel = tk.Frame(self)
        plot_view_panel.grid_columnconfigure(0, weight=1)
        plot_view_panel.grid_rowconfigure(0, weight=1)
        plot_view_panel.grid_rowconfigure(1, weight=5)
        plot_view_panel.grid(row=0, column=2, sticky="nsew")
        # Title
        self.plot_view_title = tk.Label(plot_view_panel, text="Potential Function")
        self.plot_view_title.grid(row=0, column=0, sticky="ew")

        return plot_view_panel
    
    def generate_plot(self):
        if self.control_scheme == "Olfati-Saber":
            if self.plot_canvas is None:
                self.fig_ref = plt.figure()
                ax = self.fig_ref.add_subplot(111)
                self.control_scheme.generate_plot_os(ax)
                self.plot_view_panel.plot_view_canvas = FigureCanvasTkAgg(self.fig_ref, master=self.plot_view_panel)
                self.plot_view_panel.plot_view_canvas.draw()
                self.plot_view_panel.plot_view_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
                self.plot_view_panel.plot_view_toolbar = NavigationToolbar2Tk(self.plot_view_panel.plot_view_canvas, self.plot_view_panel)
                self.plot_view_panel.plot_view_toolbar.update()
                self.plot_view_panel.plot_view_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
            else:
                self.control_scheme.generate_plot_os(self.fig_ref.get_axes()[0])
    
    def generate_plot_os(self, ax):
        ax.cla()
        r_coh = self.r_coh_param.get()
        d_ref = self.d_ref_param.get()
        a = float(self.a_param.get().strip())
        b = float(self.b_param.get().strip())
        c = float(self.c_param.get().strip())
        delta = float(self.delta_param.get().strip())
        x = np.linspace(0, r_coh, r_coh/0.01)
        w = olsaber.get_neighbour_weight(x, r_coh, delta)
        sigma = olsaber.get_cohesion_intensity(x, d_ref, a, b, c)
        ax.plot(x, w, label="Weight", color="black", linestyle='--')
        ax.plot(x, sigma, label="Cohesion", color="red", linestyle='--')
        ax.plot(x, w*sigma, label="Potential", color="blue", linestyle='-')

    def calculate_c_param(self, *args):
        if self.c_param_auto.get():
            # calculate c value
            a = float(self.a_param.get().strip())
            b = float(self.b_param.get().strip())
            c_val = (b - a)/(2*np.sqrt(a*b))
            self.c_textbox.config(state="disabled")
            self.c_param.set(str.format("{:.4f}", c_val))
        else:
            self.c_textbox.config(state="normal")

    def save_app_params(self):
        self.current_app_config["a"] = float(self.a_param.get())
        self.current_app_config["b"] = float(self.b_param.get())
        self.current_app_config["c"] = float(self.c_param.get())
        self.current_app_config["delta"] = float(self.delta_param.get())
        self.current_app_config["r_coh"] = self.r_coh_param.get()
        self.current_app_config["d_ref"] = self.d_ref_param.get()
        self.current_app_config["c_auto"] = self.c_param_auto.get()
        self.parent.update_app_params_dict(self.current_app_config)

    def on_apply(self):
        self.save_app_params()
        self.destroy()

    def on_cancel(self):
        self.destroy()



# ------------------------------- #
# --- TESTING PURPOSES ONLY ---   #
# ------------------------------- #
if __name__ == "__main__":
    root = tk.Tk()
    root.get_app_params_dict = lambda: {}
    root.update_app_params_dict = lambda a: a
    ControlSchemePanel(root, "Olfati-Saber")
    root.mainloop()