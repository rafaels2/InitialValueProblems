import numpy as np
from numpy import linalg as la
from problems.second_order.second_order import SecondOrder

np.seterr(all="raise")


class SecondOrderBackwardEuler(SecondOrder):
    def _generate_operator(self):
        operator = la.inv(self.Id - self._k * (1j * np.matmul(self.A, self.DPM) + self.C))

        def func(current_state, *_):
            return np.matmul(operator, current_state)

        return func


