import numpy as np
from numpy import linalg as la
from problems.second_order.second_order import SecondOrder

np.seterr(all="raise")


class SecondOrderDuFort(SecondOrder):
    def _generate_operator(self):
        lambda_m = 4 + 4 / (self._h ** 2)
        gamma = 1
        operator = (
            2
            * self._k
            * (
                1j * np.matmul(self.A, self.DPM)
                + self.C
                + 2 * gamma * lambda_m * self.Id
            )
        )
        fe_operator = self.Id + 0.5 * operator
        denominator = 1 + 2 * self._k * gamma * lambda_m

        def func(current_state, all_states, current_time):
            if current_state.max() > 2:
                print("wwwww")
            if len(all_states) >= 2:
                return (
                    np.matmul(operator, current_state)
                    + (1 - 2 * self._k * gamma * lambda_m) * all_states[-2]
                ) / denominator
            else:
                return np.matmul(fe_operator, current_state)

        return func
