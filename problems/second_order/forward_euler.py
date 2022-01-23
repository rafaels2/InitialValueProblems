import numpy as np
from problems.second_order.second_order import SecondOrder

np.seterr(all="raise")


class SecondOrderForwardEuler(SecondOrder):
    def _generate_operator(self):
        operator = self.Id + self._k * (1j * np.matmul(self.A, self.DPM) + self.C)

        def func(current_state, *_):
            return np.matmul(operator, current_state)

        return func


class SecondOrderModifiedForwardEuler(SecondOrder):
    SIGMA = 0.25

    def _generate_operator(self):
        operator = self.Id + self._k * (
            1j * np.matmul(self.A, self.DPM)
            - self.SIGMA * self._h ** 2 * np.matmul(self.DPM, self.DPM)
            + self.C
        )

        def func(current_state, *_):
            return np.matmul(operator, current_state)

        return func
