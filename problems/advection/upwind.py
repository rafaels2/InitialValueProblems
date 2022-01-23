import numpy as np

from problems.problem import Problem
from utils import get_d_plus, get_d_minus


class AdvectionUpwind(Problem):
    def initial_function(self, x):
        return np.cos(2 * np.pi * x)

    def solution(self, x, t):
        return self.initial_function(x + t)

    def _generate_operator(self):
        size = self._sites.shape[0]
        operator = np.eye(size) + 1 * self._k * get_d_plus(size, self._h)

        def func(current_state, *_):
            return np.matmul(operator, current_state)

        return func
