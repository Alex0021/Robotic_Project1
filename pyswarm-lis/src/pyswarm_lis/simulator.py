import numpy as np
from threading import Thread
import time

from pyswarm_lis.swarm import Swarm
from pyswarm_lis.recorder import SwarmRecorder

class Simulator():
    """
    Class implementing a simple simulator for the swarm.
    """
    def __init__(self, dt: int, swarm: Swarm, recorder: SwarmRecorder = None):
        self._dt = dt
        self.scheduler = Thread(target=self.step)
        self._swarm = swarm
        self._running = False
        self._paused = False
        self._step = False
        self._simulation_time = 0
        self.last_time = 0
        self.MAX_SPEED = False
        self._recorder = recorder

    def step(self):
        """
        Perform a single simulation step.
        """
        while(self._running):
            if (not self._paused) or self._step:
                self.last_time = time.time()
                # Update all swarm members
                self._swarm.update(self._dt)
                self._step = False
                # Record data if necessary
                if self._recorder:
                    self._recorder.record()
                self._simulation_time += self._dt
                # Check realtime speed
                if not self.MAX_SPEED:
                    t = time.time() - self.last_time
                    if t < self._dt:
                        time.sleep(self._dt - t)
            else:
                time.sleep(0.01)
                    

    def start(self):
        """
        Start or resume the simulation.
        """
        if not self._running:
            self._simulation_time = 0
            self._running = True
            self._paused = False
            self.scheduler.start()
        elif self._paused:
            self._paused = False

    def stop(self):
        """
        Stop the simulation (exit the thread).
        """
        self._running = False
    
    def pause(self):
        """
        Pause the simulation.
        """
        self._paused = True

    def paused(self) -> bool:
        """
        Check if the simulation is paused.

        Returns:
            bool: True if the simulation is paused, False otherwise
        """
        return self._paused

    def single_step(self):
        """
        Perform a single simulation step.
        """
        self._step = True

    def get_total_time(self) -> int:
        """
        Get the total simulation time.

        Returns:
            int: simulation runtime
        """
        return self._simulation_time
