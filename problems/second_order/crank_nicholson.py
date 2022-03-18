import numpy as np
from numpy import linalg as la
from problems.second_order.second_order import SecondOrder

np.seterr(all="raise")


class SecondOrderCrankNicholson(SecondOrder):
    def _generate_operator(self):
        # self._k = 0.9885 * self._k
        operator = np.matmul(
            la.inv(
                self.Id - 0.5 * self._k * (1j * np.matmul(self.A, self.DPM) + self.C)
            ),
            self.Id + 0.5 * self._k * (1j * np.matmul(self.A, self.DPM) + self.C),
        )

        def func(current_state, *_):
            return np.matmul(operator, current_state)

        return func
