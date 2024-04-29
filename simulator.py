import numpy as np
from swarm import Swarm
from threading import Thread
import time

class Simulator():
    def __init__(self, dt, swarm, update_ui_fct=None):
        self._dt = dt
        self.scheduler = Thread(target=self.step)
        self._swarm = swarm
        self._running = False
        self._paused = False
        self._step = False
        self.record_file = None
        self.record_data = False
        self.record_time = 0
        self._simulation_time = 0

    def step(self):
        '''
        Perform a simulation step
        '''
        while(self._running):
            if (not self._paused) or self._step:
                acc = np.zeros(3)
                self._swarm.update(self._dt, acc)
                self._step = False
                self._dump_data_to_file()
                self._simulation_time += self._dt
            time.sleep(self._dt)

    def start(self):
        if self._paused:
            self._paused = False
        elif not self._running:
            self._simulation_time = 0
            self._running = True
            self.scheduler.start()

    def stop(self):
        self._running = False
    
    def pause(self):
        self._paused = True

    def paused(self):
        return self._paused

    def single_step(self):
        self._step = True

    def get_total_time(self):
        return self._simulation_time

    def start_recording(self, output_file_path):
        if self.record_file is None:
            try:
                self.record_file = open(output_file_path, 'w')
                print("Recording data to file: {0}".format(output_file_path))
                # Add header to file
                self.record_file.write("sim_time,drone_count,selected_drone,noise_type,noise_dist,noise_cone,neighbor_metric,n_neighbor,dist_to_center,viewing_error,coverage\n")
            except:
                print("ERROR :: Could not open file for recording")
        self.record_data = True
        self._dump_data_to_file()

    def stop_recording(self):
        self.record_data = False
        self.record_time = 0
        if self.record_file is not None:
            self.record_file.close()
            self.record_file = None

    def _dump_data_to_file(self):
        if self.record_data:
            data = [str(self._simulation_time)]
            data.append(str(self._swarm.count))
            data.append(str(self._swarm.selected_drone))
            data.append(str(self._swarm.get_noise().get('type', '')))
            data.append(str(self._swarm.get_noise().get('param_dist', '')))
            data.append(str(self._swarm.get_noise().get('param_dir', '')))
            data.append(str(self._swarm.get_neighbor_metric("metric", "")))
            data.append(str(len(self._swarm.members[self._swarm.selected_drone].neighbors)))
            dist_to_center = np.linalg.norm(self._swarm.get_swarm_center() - self._swarm.members[self._swarm.selected_drone].pos)
            data.append(str(dist_to_center))
            data.append(str(self._swarm.members[self._swarm.selected_drone].viewing_error))
            data.append(str(self._swarm.swarm_coverage))
            self.record_file.write(','.join(data) + '\n')
            self.record_time += self._dt
