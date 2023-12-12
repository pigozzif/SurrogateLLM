import argparse
import logging
import random
import time
from multiprocessing import Pool

from cec2010.functions import *
from evo.evolution.algorithms import StochasticSolver
from evo.evolution.objectives import ObjectiveDict
from evo.listeners.listener import FileListener


def parse_args():
    parser = argparse.ArgumentParser(prog="SurrogateOptimization")
    parser.add_argument("--s", type=int, default=0, help="seeds")
    parser.add_argument("--p", type=int, default=0, help="problem")
    parser.add_argument("--solver", type=str, default="ga", help="solver")
    parser.add_argument("--n_params", type=int, default=1000, help="solution size")
    parser.add_argument("--evals", type=int, default=15000, help="fitness evaluations")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parallel_solve(solver, config, listener):
    best_result = None
    best_fitness = float("inf")
    start_time = time.time()
    evaluated = 0
    j = 0
    problem = eval(config.p)
    while evaluated < config.evals:
        solutions = solver.ask()
        before_time = time.time()
        fitness_list = [problem(x) for x in solutions]
        after_time = time.time()
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j,
                           "evaluations": evaluated,
                           "time.total": time.time() - start_time,
                           "time.model": before_time - after_time,
                           "time.eval": after_time - before_time,
                           "best.fitness": result[1]})
        if result[1] <= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
        evaluated += len(solutions)
        j += 1
    return best_result, best_fitness


def run_problem(args):
    config, s = args
    file_name = ".".join([config.solver, config.p, str(s), "txt"])
    objectives_dict = ObjectiveDict()
    objectives_dict.add_objective(name="fitness", maximize=False, best_value=0.0, worst_value=float("inf"))
    listener = FileListener(file_name=file_name, header=["iteration",
                                                         "evaluations",
                                                         "time.total",
                                                         "time.model",
                                                         "time.eval",
                                                         "best.fitness"])
    solver = StochasticSolver.create_solver(name=config.solver,
                                            seed=s,
                                            num_params=config.n_params,
                                            pop_size=20,
                                            genotype_factory="uniform_float",
                                            objectives_dict=objectives_dict,
                                            offspring_size=20,
                                            remap=False,
                                            genetic_operators={"gaussian_mut": 1.0},
                                            genotype_filter=None,
                                            tournament_size=5,
                                            mu=0.0,
                                            sigma=0.1,
                                            n=config.n_params,
                                            range=(0, 1))
    return parallel_solve(solver=solver, config=config, listener=listener)


if __name__ == "__main__":
    arguments = parse_args()
    with Pool(arguments.s) as pool:
        results = pool.map(run_problem, [(arguments, i) for i in range(arguments.s)])
