import numpy as np

from problems.problem import Problem
import utils


class SecondOrder(Problem):
    _A = np.array([[0, 1], [1, 0]])
    _C = 1j * np.array([[3, -1], [-1, 3]])
    omega = 2 * np.pi
    NDIM = _C.shape[0]

    def __init__(self, sites, h, k):
        self._size = sites.shape[0]
        self.Id = np.eye(self.NDIM * self._size)
        self.A = utils.from_point_to_array(self._A, self._size)
        self.C = utils.from_point_to_array(self._C, self._size)
        self.DPM = utils.from_point_to_array(
            utils.get_d_plus_minus(self._size, h), self.NDIM
        )
        super(SecondOrder, self).__init__(sites, h, k)

    def initial_function(self, x):
        return self.solution(x, 0)

    def solution(self, x, t):
        argument = (self.omega * x) + (1 + 4 * (self.omega ** 2)) * t
        return np.exp(3j * t) * np.array([np.cos(argument), -1j * np.sin(argument)])

    def _generate_initial_value(self):
        initial_value = np.zeros((self._size * self.NDIM,), dtype=complex)
        for index in range(self._size):
            for dim in range(self.NDIM):
                initial_value[dim * self._size + index] = self.initial_function(
                    self._sites[index]
                )[dim]
        return initial_value


def main():
    so = SecondOrder(np.array([0, 0.1, 0.2, 0.3, 0.4]), 00.1, 0.05)
    return so


if __name__ == "__main__":
    so = main()
