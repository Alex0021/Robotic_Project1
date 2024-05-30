import numpy as np
from swarm import Swarm

class SwarmRecorder():
    """
        Record the data of the swarm during the simulation. 
        If multiple runs are performed, the data is averaged over all runs.
        Structure:
        {
            "params": sim_params, 
            "selected_drone": self._swarm.selected_drone, 
            "timesteps": [], 
            "drone_data": [], 
            "swarm_data": [], 
            "timings": [],
        }
        drone_data: list of drone data for every timesteps for every drone 

        --> [# neighbors, avg neighbors dist, distance weight convex hull, viewing error, min_viewing_diff]

        swarm_data: list of swarm center and viewing coverage for every timesteps
        
    """
    def __init__(self, swarm: Swarm=None, dt:float=0.01, verbose=False, MAX_BUFFER_SIZE:int=50_000):
        self._swarm = swarm
        self._data_dict = {}
        self._current_run_dict = {}
        self._time = 0
        self.buff_idx = 0
        self._dt = dt
        self._record = False
        self._run_count = 0
        self.min_timesteps = MAX_BUFFER_SIZE
        self._verbose = verbose
        self.MAX_BUFFER_SIZE = MAX_BUFFER_SIZE

    def start(self, sim_params:dict={}, sim_time:int=0):
        if self._swarm is None:
            raise Exception("RECORDER :: Swarm object not set!")
        sim_params['sim_time'] = sim_time
        # Initialize all buffers & data structures
        NUM_DRONES = self._swarm.count
        NUM_DRONE_PARAMS = 6
        NUM_TIMINGS_PARAMS = 3
        self._current_run_dict = {
            "params": sim_params, 
            "timesteps": np.zeros(self.MAX_BUFFER_SIZE, dtype=np.float32), 
            "drone_data": np.zeros((self.MAX_BUFFER_SIZE, NUM_DRONES, NUM_DRONE_PARAMS), dtype=np.float32),
            "swarm_data": { 'centers': np.zeros((self.MAX_BUFFER_SIZE, 3), dtype=np.float32), 
                            'viewing_coverage': np.zeros(self.MAX_BUFFER_SIZE, dtype=np.float32) }, 
            "timings": np.zeros((self.MAX_BUFFER_SIZE, NUM_TIMINGS_PARAMS), dtype=np.float32),
        }
        self._record = True
        self._time = 0
        self.buff_idx = 0
        if self._verbose:
            print("RECORD DATA: ON")

    def _terminate_run(self):
        # Update record dict with running average
        self._run_count += 1
        if self._run_count == 1:
            self._record_dict = self._current_run_dict
        else:
            # Verify the minimum length of the data
            if self.buff_idx + 1 < self.min_timesteps:
                self.min_timesteps = self.buff_idx + 1
            # Update the swarm data
            for k in self._record_dict['swarm_data'].keys():
                self._record_dict['swarm_data'][k][:self.min_timesteps] = (self._record_dict['swarm_data'][k][:self.min_timesteps]*self._run_count + self._current_run_dict['swarm_data'][k][:self.min_timesteps])/(self._run_count+1)
            # Update the drone data - TODO: Find way to calculate the average of the drone data over multiple runs (drones might not be in same order due to stochasticity)
            # self._record_dict['drone_data'][:self.min_timesteps] = (self._record_dict['drone_data'][:self.min_timesteps]*self._run_count + self._current_run_dict['drone_data'][:self.min_timesteps])/(self._run_count+1)
            # Update the timings
            self._record_dict['timings'][:self.min_timesteps] = (self._record_dict['timings'][:self.min_timesteps]*self._run_count + self._current_run_dict['timings'][:self.min_timesteps])/(self._run_count+1)
        if self._verbose:
            print("RECORD DATA: RUN COUNT ==> ", self._run_count)

    def stop(self):
        self._record = False
        # Terminating the current run
        self._terminate_run()
        if self._verbose:
            print("RECORD DATA: OFF")

    def record(self):
        if self._record:
            self._current_run_dict['timesteps'][self.buff_idx] = self._time
            # Record swarm center
            swarm_center = self._swarm.get_swarm_center()
            swarm_viewing_coverage = self._swarm.swarm_coverage
            self._current_run_dict["swarm_data"]['centers'][self.buff_idx] = swarm_center
            self._current_run_dict["swarm_data"]['viewing_coverage'][self.buff_idx] = swarm_viewing_coverage
            # Record the timings
            self._current_run_dict["timings"][self.buff_idx] = np.array([self._swarm.timing_neighborhood, 
                                                                        self._swarm.timing_viewing_dir, 
                                                                        self._swarm.timing_coverage])
            # Get drone data: [# neighbors, avg neighbors dist, distance weight convex hull, viewing error, min_viewing_diff, yaw]
            data = np.zeros((self._swarm.count, 6))
            for i in range(self._swarm.count):
                drone = self._swarm.members[i]
                avg_neighbor_dst = 0
                n_count = len(drone.neighbors)
                if n_count > 0:
                    avg_neighbor_dst = np.mean([n.distance for n in drone.neighbors])
                    min_angle = np.pi
                    for n in drone.neighbors:
                        diff = abs(n.angles[2] - drone.angles[2])
                        diff = min(diff, 2*np.pi - diff)
                        if diff < min_angle:
                            min_angle = diff
                    #avg_neighbor_view_diff /= len(drone.neighbors)
                dist_weight = self._swarm.dist_weights[i]
                data[i, :] = [n_count, avg_neighbor_dst, dist_weight, drone.viewing_error, min_angle, drone.angles[2]]
        
            self._current_run_dict['drone_data'][self.buff_idx] = data
            self._time += self._dt
            self.buff_idx += 1

    def get_data(self):
        return self._record_dict

    def export(self, filepath: str):
        if self._record:
            self.stop()
        # Keep only the minimum length of the data
        self._record_dict['timesteps'] = self._record_dict['timesteps'][:self.min_timesteps]
        self._record_dict['drone_data'] = self._record_dict['drone_data'][:self.min_timesteps]
        for k in self._record_dict['swarm_data'].keys():
            self._record_dict['swarm_data'][k] =self._record_dict['swarm_data'][k][:self.min_timesteps]
        # Save data to file using pickle
        np.save(filepath, self._record_dict)
        if self._verbose:
            print("RECORD DATA: SAVED ==> Output file: ", filepath)

    def clear(self):
        self._record_dict = {}
        self._current_run_dict = {}
        self._time = 0
        self._dt = 0
        self._run_count = 0
        self._record = False