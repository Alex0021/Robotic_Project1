import tkinter as tk
import json
import threading
import time
import os
import numpy as np

# Interval to check if the simulation has finished
CHECK_INTERVAL = 0.1

class AutorunSim:
    def __init__(self, app: tk.Tk, tests_file: str):
        self._app = app
        self.run_count = 0
        self.subtest_count = 0
        self.test_count = 0
        # Load the tests description file
        try:
            with open(tests_file, 'r') as f:
                self._tests = json.load(f)
        except FileNotFoundError:
            self._tests = {}
        
    def _thread_run_all(self):
        self.nb_tests = len(self._tests)
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
            print(f"Test {self.name} not found")
            return
        test = self._tests[name]
        # Determine the number of subtests
        self.nb_subtests = np.prod([len(v['values']) for v in test['var']])
        # Determine the number of runs
        self.nb_run = test.get('repeat', 1)
        # Reset simulation and re initialize
        self._app._button_reset_callback()
        self._app._initialize_simulation(verbose=False)
        # Create output directory
        try:
            dim_text = '2D' if self._app.swarm.is_2D else '3D'
        except: 
            dim_text = ''
        self.folder_name = f'{dim_text}_{name}'
        if not os.path.exists(f'sim_results/{self.folder_name}'):
            os.makedirs(f'sim_results/{self.folder_name}')
        # Apply all parameters to the app
        for param, value in zip(test.get('metrics', []), test.get('values', [])):
            print(f"Setting {param} to {value}")
            self._app.set_var_value(param, value)
        # Hide rendering
        self._app.set_rendering('off')
        self._app.set_trajectory_mode('on')
        self.subtest_count = 0
        print(f"|== Starting test: {name} ==|")
        self.run_subtests(0, test['var'],{})
        # try:
        #     self.run_subtests(0, test['var'],{})
        # except KeyError as e:
        #     print("Value not found in test description")
        self.end_test()


    def run_subtests(self, index: int, vars: list, params: dict):
        var_name = vars[index]['name']
        values = vars[index]['values']
        if index == len(vars)-1:
            # Run tests for last variable
            for v in values:
                new_params = params.copy()
                new_params.update({var_name: v})
                self.run_step(new_params)
            return
        else:
            for v in values:
                params.update({var_name: v})
                self.run_subtests(index+1, vars, params)

    def end_test(self):
        self.test_count += 1
        print(f"!--> TEST COMPLETED: {self.test_count}/{self.nb_tests}")

    def end_subtest(self):
        self.subtest_count += 1
        self.run_count = 0
        print("Autorun step completed: {0:.1f}%".format(self.subtest_count/self.nb_subtests*100))
        self._app.stop_recording_callback()
        # Reset simulation
        self._app._button_reset_callback()
        self._app.save_recording()

    def end_step(self):
        self.run_count += 1
        if self.nb_run == 1:
            self.end_subtest()
        else:
            print("Autorun run completed: {0:.1f}%".format(self.run_count/self.nb_run*100))
            if self.run_count == self.nb_run:
                self.end_subtest()
            else:
                self._app.stop_recording_callback()
                # Reset simulation
                self._app._button_reset_callback()
                # Start next run
                self.run_step()


    def stop(self):
        self.running = False

    def run_step(self, params: dict={}):
        # Initialize sim
        self._app._initialize_simulation(verbose=False)
        # Setting var values (if needed)
        if len(params) > 0:
            for param, value in params.items():
                print(f"Setting {param} to {value}")
                if param == ('viewing_metric_algorithm'):
                    if value.startswith('convex_hull'):
                        self._app.set_var_value("viewing_metric_algorithm", "convex_hull")
                        self._app.set_var_value("viewing_metric_faces", value.split('_')[-1])
                    elif value.startswith('outer'):
                        self._app.set_var_value("viewing_metric_algorithm", "outer")
                        self._app.set_var_value("viewing_metric_outer_points", value.split('_')[-1])
                    else:
                        self._app.set_var_value("viewing_metric_algorithm", value)
                elif param == "noise_params":
                    self._app.set_var_value("noise_param_dist", value[0])
                    self._app.set_var_value("noise_param_dir", value[1])
                else:
                    self._app.set_var_value(param, value)
            # Set output file name
            description = '_'.join([f"{k}_{v}" for k, v in params.items()])
            self._app.var_output_csv.set(f'{self.folder_name}/{description}')
        self._app.sim.MAX_SPEED = True
        #self._app.start_recording_callback()
        self._app._btn_simulate_callback()
        self._app.swarm.circle_done = False
        self.running = True
        self.recording_started = False
        while self.running:
            if self._app.swarm.circle_done:
                    self.running = False
                    self.end_step()
            if self._app.swarm._stabilized and not self.recording_started:
                self._app.start_recording_callback()
                self.recording_started = True
            time.sleep(CHECK_INTERVAL)

    