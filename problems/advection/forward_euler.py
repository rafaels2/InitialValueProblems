import numpy as np

from problems.problem import Problem
from utils import get_off_diagonal_diff


class AdvectionForwardEuler(Problem):
    def initial_function(self, x):
        return np.cos(2 * np.pi * x)

    def solution(self, x, t):
        return self.initial_function(x + t)

    def _generate_operator(self):
        size = self._sites.shape[0]
        operator = np.eye(size)
        delta_coefficient = self._k / (2 * self._h)
        operator += delta_coefficient * get_off_diagonal_diff(size)

        def func(current_state, *_):
            return np.matmul(operator, current_state)

        return func
