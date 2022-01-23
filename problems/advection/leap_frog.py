import numpy as np

from problems.problem import Problem
from utils import get_off_diagonal_diff


class AdvectionLeapFrog(Problem):
    def initial_function(self, x):
        return np.cos(2 * np.pi * x)

    def solution(self, x, t):
        return self.initial_function(x + t)

    def _generate_operator(self):
        size = self._sites.shape[0]
        delta_coefficient = self._k / self._h
        off_diagonal_diff = get_off_diagonal_diff(size)
        operator = delta_coefficient * off_diagonal_diff

        def func(current_state, all_states, _):
            if len(all_states) >= 2:
                return all_states[-2] + np.matmul(operator, current_state)
            else:
                return np.matmul(
                    np.eye(size) + 0.5 * delta_coefficient * off_diagonal_diff,
                    current_state,
                )

        return func
