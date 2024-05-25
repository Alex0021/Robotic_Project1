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
        self.record_data = False
        self.record_time = 0
        self._simulation_time = 0
        self.last_time = 0
        self.MAX_SPEED = False

    def step(self):
        '''
        Perform a simulation step
        '''
        while(self._running):
            if (not self._paused) or self._step:
                self.last_time = time.time()
                acc = np.zeros(3)
                self._swarm.update(self._dt, acc)
                self._step = False
                self._dump_data_to_file()
                self._simulation_time += self._dt
                if not self.MAX_SPEED:
                    if time.time() - self.last_time < self._dt:
                        time.sleep(self._dt - (time.time() - self.last_time))
                    
            #time.sleep(self._dt)

    def start(self):
        if not self._running:
            self._simulation_time = 0
            self._running = True
            self._paused = False
            self.scheduler.start()
        elif self._paused:
            self._paused = False

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

    def start_recording(self, fixed_params=dict()):
        self.record_dict = {"params": fixed_params, "selected_drone": self._swarm.selected_drone, "timesteps": [], "drone_data": [], "swarm_center": [], "timings": []}
        self.recorded_timesteps = []
        self.recorded_data = []
        print("RECORD DATA: ON")
        self.record_data = True

    def stop_recording(self, output_file_path):
        self.record_data = False
        self.record_time = 0
        self.record_dict["timesteps"] = self.recorded_timesteps
        self.record_dict["drone_data"] = self.recorded_data
        print("RECORD DATA: OFF ==> Output file: ", output_file_path + '.npy')
        np.save(output_file_path + '.npy', self.record_dict)

    def _dump_data_to_file(self):
        if self.record_data:
            self.recorded_timesteps.append(self.record_time)
            data = []
            swarm_center = self._swarm.get_swarm_center()
            self.record_dict["swarm_center"].append(swarm_center)
            self.record_dict["timings"].append([self._swarm.timing_neighborhood, self._swarm.timing_viewing_dir, self._swarm.timing_coverage])
            for i in range(self._swarm.count):
                drone = self._swarm.members[i]
                avg_neighbor_dst = 0
                avg_neighbor_view_diff = 0                
                if len(drone.neighbors) > 0:
                    avg_neighbor_dst = np.mean([n.distance for n in drone.neighbors])
                    min_angle = np.pi
                    for n in drone.neighbors:
                        diff = abs(n.angles[2] - drone.angles[2])
                        diff = min(diff, 2*np.pi - diff)
                        if diff < min_angle:
                            min_angle = diff
                    avg_neighbor_view_diff = min_angle
                    #avg_neighbor_view_diff /= len(drone.neighbors)
                dist_weight = self._swarm.dist_weights[i]
                data.append([len(drone.neighbors), avg_neighbor_dst, dist_weight, drone.viewing_error, self._swarm.swarm_coverage, avg_neighbor_view_diff])
            self.recorded_data.append(data)
            self.record_time += self._dt
