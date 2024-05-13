import tkinter as tk
import json
import threading
import time
import os

CHECK_INTERVAL = 0.1

class AutorunSim:
    def __init__(self, app: tk.Tk, tests_file: str):
        self._app = app
        # Load the tests description file
        try:
            with open(tests_file, 'r') as f:
                self._tests = json.load(f)
        except FileNotFoundError:
            self._tests = {}
        
    def _thread_run_all(self):
        for name in self._tests.keys():
            self.run_test(name)

    def run_all(self):
        # Start the thread
        self.thread = threading.Thread(target=self._thread_run_all)
        self.running = True
        self.thread.start()

    def run_test(self, name: str):
        print(f"Preparing test: {name}")
        # Retrieve the test description
        if name not in self._tests:
            print(f"Test {name} not found")
            return
        test = self._tests[name]
        # Reset simulation and re initialize
        self._app._button_reset_callback()
        self._app._initialize_simulation(verbose=False)
        # Create output directory
        try:
            dim_text = '2D_' if self._app.swarm.is_2D else '3D_'
        except: 
            dim_text = ''
        self.autorun_filename = test.get('file_basename', 'unknown')
        self.autorun_filename = dim_text + self.autorun_filename
        if not os.path.exists(f'output/{self.autorun_filename}'):
            os.makedirs(f'output/{self.autorun_filename}')
        # Apply all parameters to the app
        for param, value in zip(test.get('metrics', []), test.get('values', [])):
            print(f"Setting {param} to {value}")
            self._app.set_var_value(param, value)
        # Retrieve varying param
        self.autorun_var_name = test.get('var', 'None')
        self.autorun_from = test.get('from', 0)
        self.autorun_to = test.get('to', 0)
        self.autorun_steps = test.get('steps', 0)
        self.var_autosim = self._app.get_var_ref(self.autorun_var_name)
        self._app.var_output_csv.set(f'{self.autorun_filename}/{self.autorun_filename}_{self.autorun_var_name}_{self.autorun_from}')
        # Start recording
        self._app.start_recording_callback()
        # Start simulation
        self._app._btn_simulate_callback()
        print("Starting autorun simulation")
        self.var_autosim.set(self.autorun_from)
        self.running = True
        self.running_step()


    def end_test(self):
        self._app._btn_pause_callback()
        self._app.stop_recording_callback()
        print("Autorun simulation completed!")

    def end_step(self):
        completion = (self.var_autosim.get() - self.autorun_from + 1)/(self.autorun_steps*(self.autorun_to-self.autorun_from + 1))*100
        print("Autorun step completed: {0:.2f}%".format(completion))
        self._app.stop_recording_callback()
        self.var_autosim.set(self.var_autosim.get() + self.autorun_steps)
        self._app.var_output_csv.set(f'{self.autorun_filename}/{self.autorun_filename}_{self.autorun_var_name}_{self.var_autosim.get()}')
        # Reset simulation and re initialize
        self._app._button_reset_callback()
        self._app._initialize_simulation(verbose=False)
        self._app.start_recording_callback()
        self._app._btn_simulate_callback()
        self._app.swarm.circle_done = False

    def stop(self):
        self.running = False

    def running_step(self):
        while self.running:
            if self._app.swarm.circle_done:
                if self.var_autosim.get() >= self.autorun_to:
                    self.end_test()
                    self.running = False
                else:
                    self.end_step()
            time.sleep(CHECK_INTERVAL)

    