import random

import numpy as np

from evo.evolution.algorithms import GPGO, GeneticAlgorithm, RandomSearch
from evo.evolution.objectives import ObjectiveDict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def create_solver(s, config, bounds=(-5, 5)):
    objectives_dict = ObjectiveDict()
    objectives_dict.add_objective(name="fitness", maximize=False, best_value=0.0, worst_value=float("inf"))
    if config.solver == "rs":
        return RandomSearch(seed=s,
                            num_params=config.num_params,
                            objectives_dict=objectives_dict,
                            range=bounds)
    elif config.solver == "ga":
        return GeneticAlgorithm(seed=s,
                                num_params=config.n_params,
                                pop_size=20,
                                genotype_factory="uniform_float",
                                objectives_dict=objectives_dict,
                                survival_selector="worst",
                                parent_selector="tournament",
                                offspring_size=20,
                                overlapping=True,
                                remap=False,
                                genetic_operators={"gaussian_mut": 1.0},
                                genotype_filter=None,
                                tournament_size=5,
                                mu=0.0,
                                sigma=0.1,
                                n=config.n_params,
                                range=(-5.0, 5.0))
    elif config.solver == "gpgo":
        return GPGO(seed=s,
                    num_params=config.n_params,
                    f=eval(config.p),
                    range=bounds)
    raise ValueError("Invalid solver name: {}".format(config.solver))