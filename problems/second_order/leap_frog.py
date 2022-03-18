import numpy as np
from numpy import linalg as la
from problems.second_order.second_order import SecondOrder

np.seterr(all="raise")


class SecondOrderLeapFrog(SecondOrder):
    def _generate_operator(self):
        operator = 2 * self._k * (1j * np.matmul(self.A, self.DPM) + self.C)
        fe_operator = self.Id + 0.5 * operator

        def func(current_state, all_states, current_time):
            if len(all_states) >= 2:
                return np.matmul(operator, current_state) + all_states[-2]
            else:
                return np.matmul(fe_operator, current_state)

        return func
