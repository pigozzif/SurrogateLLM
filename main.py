import argparse
import logging
import time
from multiprocessing import Pool

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
    set_seed(s)
    file_name = ".".join([config.solver, config.p, str(s), "txt"])
    listener = FileListener(file_name=file_name, header=["iteration",
                                                         "evaluations",
                                                         "time.total",
                                                         "time.model",
                                                         "time.eval",
                                                         "best.fitness"])
    solver = create_solver(s=s, config=config)
    return parallel_solve(solver=solver, config=config, listener=listener)


if __name__ == "__main__":
    arguments = parse_args()
    with Pool(arguments.s) as pool:
        results = pool.map(run_problem, [(arguments, i) for i in range(arguments.s)])


# def run_problems(args):
#     pidx, config = args
    # config.s = pidx
#     set_seed(config.s)
#     if pidx == 0:
#         os.system("python3 run_experiment.py --max-eval={0} --seed={1} windwake --file=expensiveoptimbenchmark/problems/example_input_windwake.json {2}".format(config.evals, config.s, config.solver))
#    elif pidx == 1:
#         os.system("python3 run_experiment.py --max-eval={0} --seed={1} esp {2}".format(config.evals, config.s, config.solver))
#     elif pidx == 2:
#         os.system("python3 run_experiment.py --max-eval={0} --seed={1} pitzdaily {2}".format(config.evals, config.s, config.solver))
#     elif pidx == 3:
#         os.system("python3 run_experiment.py --max-eval={0} --seed={1} hpo --folder=expensiveoptimbenchmark/problems/steel+plates+faults {2}".format(config.evals, config.s, config.solver))
    # os.system("python run_experiment.py --max-eval={0} --seed={1} rosenbrock --n-cont=10 {2}".format(config.evals, config.s, config.solver))
    # os.system("python run_experiment.py --max-eval={0} --seed={1} rosen {2}".format(config.s, config.solver))
    # os.system("python run_experiment.py --max-eval={0} --seed={1} convex -d=1000 {2}".format(config.evals, config.s, config.solver))


# if __name__ == "__main__":
#     args = parse_args()
#     pids = [int(pid) for pid in args.pids.split("-")]
    # with Pool(5) as pool:
    #     results = pool.map(run_problems, [(i, args) for i in range(5)])
#      for pidx in pids:
#         run_problems((pidx, args))
