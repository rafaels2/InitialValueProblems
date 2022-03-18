import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "results_2"


def get_off_diagonal_diff(size):
    operator = np.zeros((size, size))

    for index in np.ndindex(size, size):
        if (index[0] - index[1]) % size == 1:
            operator[index] = -1
        elif (index[1] - index[0]) % size == 1:
            operator[index] = 1

    return operator


def get_off_diagonal_quadratic_diff(size):
    operator = np.zeros((size, size))

    for index in np.ndindex(size, size):
        if (index[0] - index[1]) % size == 1 or (index[1] - index[0]) % size == 1:
            operator[index] = 1
        elif index[0] == index[1]:
            operator[index] = -2

    return operator


def get_e(size, order=1):
    operator = np.zeros((size, size))
    order = order % size

    for index in np.ndindex(size, size):
        if (index[1] - index[0]) % size == order:
            operator[index] = 1

    return operator


def get_d_plus(size, fill_distance):
    return (get_e(size, 1) - get_e(size, 0)) / fill_distance


def get_d_minus(size, fill_distance):
    return (get_e(size, 0) - get_e(size, -1)) / fill_distance


def get_d_plus_minus(size, fill_distance):
    return np.matmul(get_d_plus(size, fill_distance), get_d_minus(size, fill_distance))


def from_1d_to_nd(operator, ndim):
    zero = np.zeros_like(operator)
    return np.block(
        [[(operator if i == j else zero) for i in range(ndim)] for j in range(ndim)]
    )


def from_point_to_array(operator, size):
    return np.block(
        [
            [np.eye(size) * operator[j, i] for i in range(operator.shape[1])]
            for j in range(operator.shape[0])
        ]
    )


def show_matrix(mat, name):
    # print(f"name: {name} shape: {mat.shape}")
    return
    plt.figure()
    plt.matshow(mat.real)
    plt.title(name)
    plt.colorbar()
    plt.show()
