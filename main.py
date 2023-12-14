import argparse
import os
import random

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="random seed")
    parser.add_argument("--pids", type=str, default="0-1", help="problem ids")
    parser.add_argument("--solver", type=str, default="randomsearch", help="solver")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_problems(args):
    pidx, config = args
    # config.s = pidx
    set_seed(config.s)
    if pidx == 0:
        os.system("python3 run_experiment.py --max-eval=1000 --seed={0} windwake --file=expensiveoptimbenchmark/problems/example_input_windwake.json {1}".format(config.s, config.solver))
    elif pidx == 1:
        os.system("python3 run_experiment.py --max-eval=1000 --seed={0} esp {1}".format(config.s, config.solver))
    elif pidx == 2:
        os.system("python3 run_experiment.py --max-eval=1000 --seed={0} pitzdaily {1}".format(config.s, config.solver))
    elif pidx == 3:
        os.system("python3 run_experiment.py --max-eval=1000 --seed={0} hpo --folder=expensiveoptimbenchmark/problems/steel+plates+faults {1}".format(config.s, config.solver))
    # os.system("python run_experiment.py --max-eval=1000 --seed={0} rosenbrock --n-cont=10 {1}".format(config.s, config.solver))
    # os.system("python run_experiment.py --max-eval=1000 --seed={0} rosen {1}".format(config.s, config.solver))
    # os.system("python run_experiment.py --max-eval=1000 --seed={0} convex -d=1000 {1}".format(config.s, config.solver))


if __name__ == "__main__":
    args = parse_args()
    pids = [int(pid) for pid in args.pids.split("-")]
    # with Pool(5) as pool:
    #     results = pool.map(run_problems, [(i, args) for i in range(5)])
    for pidx in pids:
        run_problems((pidx, args))
