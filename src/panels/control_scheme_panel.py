import tkinter as tk
from tkinter import ttk
import numpy as np

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

        self.control_scheme = current_control_scheme

        self.init_common_components()

    def init_common_components(self):
        self.title("Control Scheme")
        self.geometry("600x300")

        # Set geometry manager
        self.grid_columnconfigure((0,1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set panel for algorithm selection
        self.algorithm_panel = tk.Frame(self)
        self.algorithm_panel.grid_columnconfigure((0,1,2), weight=1)
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
        self.parameters_panel.grid(row=2, column=0, columnspan=2, sticky="nsew")

    def init_parameters_panel(self) -> tk.Frame:
        if self.control_scheme == "Olfati-Saber":
            parameters_panel = tk.Frame(self.algorithm_panel)
            parameters_panel.grid_columnconfigure((0,1), weight=1)
            parameters_panel.grid_rowconfigure(0, weight=1)
            # d_ref parameter
            self.d_ref_label = tk.Label(parameters_panel, text="d_ref: ")
            self.d_ref_label.grid(row=0, column=0, sticky="w", padx=5)
            self.d_ref_slider = ttk.Scale(parameters_panel, from_=D_REF_LIMITS[0], to=D_REF_LIMITS[1], orient="horizontal")
            self.d_ref_slider.set(self.current_app_config.get("d_ref", np.mean(D_REF_LIMITS)))
            self.d_ref_slider.grid(row=0, column=1, sticky="ew", padx=5)
            # r_coh parameter
            self.r_coh_label = tk.Label(parameters_panel, text="r_coh: ")
            self.r_coh_label.grid(row=1, column=0, sticky="w", padx=5)
            self.r_coh_spinner = ttk.Spinbox(parameters_panel, from_=R_COH_LIMITS[0], to=R_COH_LIMITS[1], increment=0.1)
            self.r_coh_spinner.insert(0, self.current_app_config.get("r_coh", 20.0))
            self.r_coh_spinner.grid(row=1, column=1, sticky="ew", padx=5)
            # delta parameter
            self.delta_label = tk.Label(parameters_panel, text="delta: ")
            self.delta_label.grid(row=2, column=0, sticky="w", padx=5)
            self.delta_textbox = tk.Entry(parameters_panel)
            self.delta_textbox.insert(0, self.current_app_config.get("delta", 1.0))
            self.delta_textbox.grid(row=2, column=1, sticky="ew", padx=5)
            # a parameter
            self.a_label = tk.Label(parameters_panel, text="a: ")
            self.a_label.grid(row=3, column=0, sticky="w", padx=5)
            self.a_textbox = tk.Entry(parameters_panel)
            self.a_textbox.insert(0, self.current_app_config.get("a", 1.0))
            self.a_textbox.grid(row=3, column=1, sticky="ew", padx=5)
            # b parameter
            self.b_label = tk.Label(parameters_panel, text="b: ")
            self.b_label.grid(row=4, column=0, sticky="w", padx=5)
            self.b_textbox = tk.Entry(parameters_panel)
            self.b_textbox.insert(0, self.current_app_config.get("b", 1.0))
            self.b_textbox.grid(row=4, column=1, sticky="ew", padx=5)
            # c parameter panel
            self.c_param_panel = tk.Frame(parameters_panel)
            self.c_param_panel.grid_columnconfigure((0,2), weight=1)
            self.c_param_panel.grid_columnconfigure(1, weight=1)
            self.c_param_panel.grid_rowconfigure(0, weight=1)
            self.c_param_panel.grid(row=5, column=0, columnspan=2, sticky="nsew")
            self.c_label = tk.Label(self.c_param_panel, text="c: ")
            self.c_label.grid(row=0, column=0, sticky="w", padx=5)
            self.c_textbox = tk.Entry(self.c_param_panel)
            self.c_textbox.insert(0, self.current_app_config.get("c", 1.0))
            self.c_textbox.grid(row=0, column=1, sticky="ew", padx=5)
            self.c_auto_checkbox = tk.Checkbutton(self.c_param_panel, text="Auto", variable=tk.BooleanVar())
            self.c_auto_checkbox.grid(row=0, column=2, sticky="ew", padx=5)
            




        



# ------------------------------- #
# --- TESTING PURPOSES ONLY ---   #
# ------------------------------- #
if __name__ == "__main__":
    root = tk.Tk()
    ControlSchemePanel(root, "Olfati-Saber")
    root.mainloop()