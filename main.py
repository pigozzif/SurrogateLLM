import argparse
import os
import random

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--solver", type=str, default="randomsearch", help="solver")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.s)
    # os.system("python run_experiment.py --max-eval=1000 windwake --file=EXPObench/expensiveoptimbenchmark/problems/example_input_windwake.json {}".format(args.solver))
    # os.system("python run_experiment.py --max-eval=1000 esp {}".format(args.solver))
    # os.system("python run_experiment.py --max-eval=1000 pitzdaily {}".format(args.solver))
    os.system("python run_experiment.py --max-eval=1000 hpo {}".format(args.solver))
