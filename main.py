import argparse
import os
import random
from multiprocessing import Pool

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=5, help="number of seeds")
    parser.add_argument("--solver", type=str, default="randomsearch", help="solver")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_problems(args):
    s, solver = args
    set_seed(s)
    # os.system("python run_experiment.py --max-eval=1000 windwake --file=EXPObench/expensiveoptimbenchmark/problems/example_input_windwake.json {}".format(solver))
    # os.system("python run_experiment.py --max-eval=1000 esp {}".format(solver))
    # os.system("python run_experiment.py --max-eval=1000 pitzdaily {}".format(solver))
    # os.system("python run_experiment.py --max-eval=1000 hpo {}".format(solver))
    os.system("python run_experiment.py --max-eval=1000 --seed={0} rosenbrock --n-cont=10 {1}".format(s, solver))
    os.system("python run_experiment.py --max-eval=1000 --seed={0} rosen {1}".format(s, solver))


if __name__ == "__main__":
    args = parse_args()
    with Pool(args.s) as pool:
        results = pool.map(run_problems, [(i, args.solver) for i in range(args.s)])
