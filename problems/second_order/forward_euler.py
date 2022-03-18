import numpy as np
from problems.second_order.second_order import SecondOrder
import matplotlib.pyplot as plt

from utils import show_matrix

np.seterr(all="raise")


class SecondOrderForwardEuler(SecondOrder):
    def _generate_operator(self):
        operator = self.Id + self._k * (1j * np.matmul(self.A, self.DPM) + self.C)

        def func(current_state, *_):
            return np.matmul(operator, current_state)

        return func


class SecondOrderModifiedForwardEuler(SecondOrder):
    def __init__(self, sites, h, k, config):
        self.SIGMA = config["sigma"]
        super(SecondOrderModifiedForwardEuler, self).__init__(sites, h, k, config)

    @property
    def special_config(self):
        return f"sigma_{self.SIGMA}"

    def _generate_operator(self):
        operator = self.Id + self._k * (
            1j * np.matmul(self.A, self.DPM)
            - self.SIGMA * self._h ** 2 * np.matmul(self.DPM, self.DPM)
            + self.C
        )
        show_matrix(operator, "operator")
        show_matrix(np.power(operator, 10), "op^10")

        def func(current_state, _, t):
            current_state_dbg = np.block(
                [
                    [current_state[::2].real],
                    [current_state[::2].imag],
                    [current_state[1::2].real],
                    [current_state[1::2].imag],
                ]
            )
            return np.matmul(operator, current_state)

        return func
