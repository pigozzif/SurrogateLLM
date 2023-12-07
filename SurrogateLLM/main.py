import argparse
from multiprocessing import Pool
import random
import logging
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--p", type=str, default="clock", help="problem")
    parser.add_argument("--np", type=int, default=1, help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="afpo", help="solver")
    parser.add_argument("--evals", type=int, default=1000, help="fitness evaluations")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)


def parallel_solve(solver, config, listener):
    best_result = None
    best_fitness = float("-inf")
    start_time = time.time()
    evaluated = 0
    j = 0
    while evaluated < config.evals:
        solutions = solver.ask()
        with Pool(config.np) as pool:
            results = pool.map(parallel_wrapper, [(config, solutions[i], i) for i in range(solver.pop_size)])
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                           "evaluations": evaluated, "best.fitness": result[1],
                           "best.solution": result[0]})
        if result[1] >= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
        evaluated += len(solutions)
        j += 1
    return best_result, best_fitness


def parallel_wrapper(arg):
    c, solution, i = arg
    fitness = evaluate_fitness(config=c, solution=solution)
    return i, -fitness


def evaluate_fitness(config, solution):
    world = Lattice.create_lattice(name=config.p,
                                   w=config.w,
                                   h=config.h,
                                   r=config.r,
                                   dt=config.dt,
                                   max_t=config.t,
                                   video_name=video_name)
    world.set_params(params=solution)
    world.solve()
    # fitness = world.get_fitness()
    return 0.0  # fitness


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.s)
