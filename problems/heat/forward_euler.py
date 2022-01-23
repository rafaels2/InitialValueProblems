import numpy as np

from problems.problem import Problem
from utils import get_off_diagonal_quadratic_diff

np.seterr(all="raise")


class HeatForwardEuler(Problem):
    def initial_function(self, x):
        return np.cos(2 * np.pi * x)

    def solution(self, x, t):
        return self.initial_function(x + t)

    def extern_function(self, t):
        argument = 2 * np.pi * (self._sites + t)
        return 4 * (np.pi ** 2) * np.cos(argument) - 2 * np.pi * np.sin(argument)

    def _generate_operator(self):
        size = self._sites.shape[0]
        operator = np.eye(size)
        delta_coefficient = self._k / (self._h ** 2)
        operator += delta_coefficient * get_off_diagonal_quadratic_diff(size)

        def func(current_state, _, current_time):
            if np.linalg.norm(current_state) > 20:
                print("hi")
            try:
                return np.matmul(
                    operator, current_state
                ) + self._k * self.extern_function(current_time)
            except FloatingPointError:
                print("hi")

        return func
