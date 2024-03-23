import numpy as np
from swarm import Swarm
from threading import Thread
import time

class Simulator():
    def __init__(self, dt, swarm):
        self._dt = dt
        self.scheduler = Thread(target=self.step)
        self._swarm = swarm
        self._running = False
        self._paused = False
        self._step = False

    def step(self):
        '''
        Perform a simulation step
        '''
        while(self._running):
            if (not self._paused) or self._step:
                acc = np.zeros(3)
                self._swarm.update(self._dt, acc)
                self._step = False
            time.sleep(self._dt)

    def start(self):
        if self._paused:
            self._paused = False
        elif not self._running:
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
