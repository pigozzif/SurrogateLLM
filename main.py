import argparse
import os
import random

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="random seed")
    parser.add_argument("--pids", type=str, default="0-1", help="problem ids")
    parser.add_argument("--solver", type=str, default="randomsearch", help="solver")
    parser.add_argument("--evals", type=int, default=1000, help="maximum evaluations")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_problems(args):
    pidx, config = args
    # config.s = pidx
    set_seed(config.s)
    if pidx == 0:
        os.system("python3 run_experiment.py --max-eval={0} --seed={1} windwake --file=expensiveoptimbenchmark/problems/example_input_windwake.json {2}".format(config.evals, config.s, config.solver))
    elif pidx == 1:
        os.system("python3 run_experiment.py --max-eval={0} --seed={1} esp {2}".format(config.evals, config.s, config.solver))
    elif pidx == 2:
        os.system("python3 run_experiment.py --max-eval={0} --seed={1} pitzdaily {2}".format(config.evals, config.s, config.solver))
    elif pidx == 3:
        os.system("python3 run_experiment.py --max-eval={0} --seed={1} hpo --folder=expensiveoptimbenchmark/problems/steel+plates+faults {2}".format(config.evals, config.s, config.solver))
    # os.system("python run_experiment.py --max-eval={0} --seed={1} rosenbrock --n-cont=10 {2}".format(config.evals, config.s, config.solver))
    # os.system("python run_experiment.py --max-eval={0} --seed={1} rosen {2}".format(config.s, config.solver))
    # os.system("python run_experiment.py --max-eval={0} --seed={1} convex -d=1000 {2}".format(config.evals, config.s, config.solver))


if __name__ == "__main__":
    args = parse_args()
    pids = [int(pid) for pid in args.pids.split("-")]
    # with Pool(5) as pool:
    #     results = pool.map(run_problems, [(i, args) for i in range(5)])
    for pidx in pids:
        run_problems((pidx, args))
