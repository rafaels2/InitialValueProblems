"""
This is the main file of the project.
It runs different scheme in different configurations, and evaluates their errors.

It was tested in Pycharm on Windows.
"""
import os
import sys

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import toml
import logging

from finite_differences import FiniteDifferences
from fit import fit_linear
from problems.second_order import (
    SecondOrderForwardEuler,
    SecondOrderModifiedForwardEuler,
    SecondOrderBackwardEuler,
    SecondOrderCrankNicholson,
    SecondOrderLeapFrog,
    SecondOrderDuFort,
)
from utils import RESULTS_PATH

NDIM = 2

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s: %(message)s"
)


def get_time_step(fill_distance, config):
    power = config["k_h_power"]
    return config["ratio"] * (fill_distance ** power)


def get_config(filename="config.toml"):
    config = toml.load(os.path.join("config", filename))
    return config


def evaluate_on_sites(sites, solution, t):
    # print("sites: ", sites[-1])
    evaluation = np.array(
        [solution(sites[index], t) for index in np.ndindex(sites.shape)]
    )
    return np.ravel(evaluation)


def evaluate_error(numerical, analytical, fill_distance):
    """
    Evaluating the error between numerical and analytical solutions.
    :param numerical:
    :param analytical:
    :param fill_distance: Required for norm_h calculation
    :return:
    """
    # print("shape: ", numerical.shape)
    if NDIM > 1:
        numerical.resize(int(numerical.shape[0] / NDIM), NDIM)
        analytical.resize(int(analytical.shape[0] / NDIM), NDIM)
    differences = la.norm(numerical - analytical, axis=1)
    error = np.sqrt(np.sum((np.abs(differences) ** 2) * fill_distance))
    plt.figure()
    plt.plot(numerical.real[:, 0], label="real(numerical[0])")
    plt.plot(analytical.real[:, 0], label="real(analytical[0])")
    plt.legend()
    plt.title(f"re 0 n = {numerical.shape[0]}, error = {error}")
    plt.show(block=False)
    plt.figure()
    plt.plot(numerical.real[:, 1], label="real(numerical[1])")
    plt.plot(analytical.real[:, 1], label="real(analytical[1])")
    plt.legend()
    plt.title(f"re 1 n = {numerical.shape[0]}, error = {error}")
    plt.show(block=False)
    plt.figure()
    plt.plot(numerical.imag[:, 0], label="imag(Numerical[0])")
    plt.plot(analytical.imag[:, 0], label="imag(Analytical[0])")
    plt.legend()
    plt.title(f"Im 0 N = {numerical.shape[0]}, error = {error}")
    plt.show(block=False)
    plt.figure()
    plt.plot(numerical.imag[:, 1], label="imag(Numerical[1])")
    plt.plot(analytical.imag[:, 1], label="imag(Analytical[1])")
    plt.legend()
    plt.title(f"Im 1 N = {numerical.shape[0]}, error = {error}")
    plt.show(block=True)
    logging.info(f"N={numerical.shape[0]}")
    return error


def run_problem(problem_type: type, config):
    errors = list()
    for number_of_sites in config["number_of_sites"]:
        # print("start")
        logging.info("============New cycle=============")
        sites = np.linspace(0, 1, number_of_sites, endpoint=False)
        fill_distance = sites[1] - sites[0]
        time_step = get_time_step(fill_distance, config)
        logging.info(f"h = {fill_distance}, k = {time_step}")
        problem = problem_type(sites, fill_distance, time_step, config)
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
        # print("t_f", engine.time)
        # solution = evaluate_on_sites(sites, problem.solution, 1.0051)
        solution = evaluate_on_sites(sites, problem.solution, config["t_f"])
        error = evaluate_error(final_state, solution, fill_distance)
        logging.info(f"Error: {error}")
        errors.append(error)
    fit_linear(
        np.log(np.array(config["number_of_sites"])) / np.log(10),
        np.log(errors) / np.log(10),
        f"{config['name']}_lambda_{config['ratio']}_power_{'k_h_power'}_{problem.special_config}",
    )


def run():
    """
    This is the main runner of the problems.
    It configures which scheme to run, and in what configuration.
    """

    # To run a single experiment, You can comment out the others.
    problems = [
        # (SecondOrderForwardEuler, "so_forward_euler_2.toml"),
        # To run the modified FE, You have to configure sigma
        # (SecondOrderModifiedForwardEuler, "so_forward_euler_mod_2.toml"),
        # (SecondOrderBackwardEuler, "so_backward_euler_2.toml"),
        # (SecondOrderLeapFrog, "so_leap_frog.toml"),
        # (SecondOrderCrankNicholson, "so_crank_2.toml"),
        (SecondOrderDuFort, "so_du_fort.toml"),
    ]

    for problem, config_file in problems:
        config = get_config(config_file)
        logging.info(f"=================Running {config['name']}===============")
        try:
            run_problem(problem, config)
        except Exception as e:
            logging.info(f"Error in run {config['name']}, with exception {e}")


def main():
    path = RESULTS_PATH
    if not os.path.isdir(path):
        os.mkdir(path)

    run()


if __name__ == "__main__":
    main()
    input("Done?")
