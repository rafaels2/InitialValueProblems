import numpy as np
from matplotlib import pyplot as plt


class FiniteDifferences(object):
    def __init__(self, fill_distance, time_step, operator, initial_value):
        self._fill_distance = fill_distance
        self._time_step = time_step
        self._operator = operator
        self._initial_value = initial_value
        self._current_state = self._initial_value
        self._current_time = 0
        self._all_states = list()
        self.last_plot = 0

    def tick(self):
        self._all_states.append(self._current_state)
        self._current_state = self._operator(
            self._current_state, self._all_states, self._current_time
        )
        self._current_time += self._time_step
        # if self._current_time - self.last_plot > 0.05:
        # plt.plot(self._current_state)
        # plt.title(f"Time: {self._current_time}")
        # plt.show()
        self.last_plot = self._current_time

    def run(self, final_time):
        while self._current_time < final_time + self._time_step:
            self.tick()
        # pass

    @property
    def state(self):
        return self._current_state

    @property
    def time(self):
        return self._current_time
