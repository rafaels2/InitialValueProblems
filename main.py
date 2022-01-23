import os

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import toml

from problems.advection import AdvectionForwardEuler, AdvectionLeapFrog, AdvectionUpwind
from finite_differences import FiniteDifferences
from fit import fit_linear
from problems.heat import HeatForwardEuler, HeatLeapFrog
from problems.second_order import SecondOrderForwardEuler, SecondOrderModifiedForwardEuler, SecondOrderBackwardEuler, \
    SecondOrderCrankNicholson

NDIM = 2


def get_time_step(fill_distance, config):
    power = config["k_h_power"]
    return config["ratio"] * (fill_distance ** power)


def get_config(filename="config.toml"):
    config = toml.load(os.path.join("config", filename))
    return config


def evaluate_on_sites(sites, solution, t):
    evaluation = np.array(
        [solution(sites[index], t) for index in np.ndindex(sites.shape)]
    )
    return np.ravel(np.transpose(evaluation))


def evaluate_error(numerical, analytical, fill_distance):
    if NDIM > 1:
        numerical.resize(NDIM, int(numerical.shape[0] / NDIM))
        analytical.resize(NDIM, int(analytical.shape[0] / NDIM))
    differences = la.norm(numerical - analytical, axis=0)
    error = np.sqrt(np.sum((np.abs(differences) ** 2) * fill_distance))
    plt.figure()
    plt.plot(la.norm(numerical, axis=0), label="abs(Numerical)")
    plt.plot(la.norm(analytical, axis=0), label="abs(Analytical)")
    plt.title(f"N = {numerical.shape[1]}, error = {error}")
    plt.show()
    return error


def run_problem(problem_type: type, config):
    errors = list()
    for number_of_sites in config["number_of_sites"]:
        print("start")
        sites = np.linspace(0, 1, number_of_sites, endpoint=False)
        fill_distance = sites[1] - sites[0]
        time_step = get_time_step(fill_distance, config)
        print(f"h = {fill_distance}, k = {time_step}")
        problem = problem_type(sites, fill_distance, time_step)
        general_initial_function = lambda x, t: problem.initial_function(x)
        initial_evaluation = evaluate_on_sites(sites, general_initial_function, 0)
        engine = FiniteDifferences(
            fill_distance,
            time_step,
            problem.operator,
            initial_evaluation,
        )
        engine.run(config["t_f"])
        final_state = engine.state
        solution = evaluate_on_sites(sites, problem.solution, engine.time)
        error = evaluate_error(final_state, solution, fill_distance)
        errors.append(error)
    fit_linear(
        np.log(np.array(config["number_of_sites"])) / np.log(10),
        np.log(errors) / np.log(10),
        problem_type.__name__,
    )


def run():
    problems = [
        # (AdvectionForwardEuler, "advection_forward_config.toml"),
        # (AdvectionForwardEuler, "advection_forward_config_2.toml"),
        # (AdvectionLeapFrog, "advection_leap_frog.toml"),
        # (AdvectionLeapFrog, "advection_leap_frog_2.toml"),
        # (HeatForwardEuler, "heat_forward_euler.toml"),
        # (HeatForwardEuler, "heat_forward_euler_2.toml"),
        # (HeatLeapFrog, "heat_leap_frog.toml"),
        # (HeatLeapFrog, "heat_leap_frog_2.toml"),
        # (AdvectionUpwind, "advection_upwind.toml"),
        # (SecondOrderModifiedForwardEuler, "so_forward_euler_2.toml"),
        # (SecondOrderBackwardEuler, "so_backward_euler_2.toml"),
        (SecondOrderCrankNicholson, "so_crank_2.toml"),
    ]

    for problem, config_file in problems:
        config = get_config(config_file)
        run_problem(problem, config)


def main():
    run()


if __name__ == "__main__":
    main()
