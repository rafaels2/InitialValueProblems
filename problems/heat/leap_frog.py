import numpy as np

from problems.problem import Problem
from utils import get_off_diagonal_quadratic_diff


class HeatLeapFrog(Problem):
    def initial_function(self, x):
        return np.cos(2 * np.pi * x)

    def solution(self, x, t):
        return self.initial_function(x + t)

    def extern_function(self, t):
        argument = 2 * np.pi * (self._sites + t)
        return 4 * (np.pi ** 2) * np.cos(argument) - 2 * np.pi * np.sin(argument)

    def _generate_operator(self):
        size = self._sites.shape[0]
        operator = np.zeros((size, size))
        delta_coefficient = 2 * self._k / (self._h ** 2)
        off_diagonal_diff = get_off_diagonal_quadratic_diff(size)
        operator += delta_coefficient * off_diagonal_diff

        def func(current_state, all_states, current_time):
            if np.linalg.norm(current_state) > 20:
                print(f"hi, {current_time}")
            if len(all_states) >= 2:
                return (
                    all_states[-2]
                    + np.matmul(operator, current_state)
                    + 2 * self._k * self.extern_function(current_time)
                )
            else:
                return (
                    np.matmul(
                        np.eye(size) + 0.5 * delta_coefficient * off_diagonal_diff,
                        current_state,
                    )
                    + self._k * self.extern_function(current_time)
                )

        return func
