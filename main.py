import argparse
import logging
import time
from multiprocessing import Pool
import os

from cec2010.functions import *
from evo.listeners.listener import FileListener
from utils import set_seed, create_solver


def parse_args():
    parser = argparse.ArgumentParser(prog="SurrogateOptimization")
    parser.add_argument("--s", type=int, default=1, help="seeds")
    parser.add_argument("--p", type=str, default="sphere", help="problem")
    parser.add_argument("--solver", type=str, default="ga", help="solver")
    parser.add_argument("--n_params", type=int, default=10, help="solution size")
    parser.add_argument("--evals", type=int, default=15000, help="fitness evaluations")
    return parser.parse_args()


def parallel_solve(solver, config, listener):
    best_fitness = float("inf")
    start_time = time.time()
    evaluated = 0
    j = 0
    while evaluated < config.evals:
        solver.solve()
        result = solver.result()  # first element is the best solution, second element is the best fitness
        evaluated = solver.get_num_evaluated()
        listener.listen(**{"iteration": j,
                           "evaluations": evaluated,
                           "time.total": time.time() - start_time,
                           "best.fitness": result[1]})
        if result[1] <= best_fitness:
            best_fitness = result[1]
        j += 1
    return best_fitness


def run_problem(args):
    config, s = args
    set_seed(s)
    file_name = os.path.join("results", ".".join([config.solver, config.p, str(s), "txt"]))
    listener = FileListener(file_name=file_name, header=["iteration",
                                                         "evaluations",
                                                         "time.total",
                                                         "best.fitness"])
    solver = create_solver(s=s, config=config)
    return parallel_solve(solver=solver, config=config, listener=listener)


if __name__ == "__main__":
    arguments = parse_args()
    with Pool(arguments.s) as pool:
        results = pool.map(run_problem, [(arguments, i) for i in range(arguments.s)])
