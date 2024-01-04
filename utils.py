import random

from cec2010.functions import *
from evo.evolution.algorithms import GPGO, GeneticAlgorithm, RandomSearch, TPE, BO
from evo.evolution.objectives import ObjectiveDict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def create_solver(s, config, bounds=(-5, 5)):
    objectives_dict = ObjectiveDict()
    objectives_dict.add_objective(name="fitness", maximize=False, best_value=0.0, worst_value=float("inf"))
    problem = eval(config.p)
    if config.solver == "rs":
        return RandomSearch(seed=s,
                            num_params=config.n_params,
                            objectives_dict=objectives_dict,
                            problem=problem,
                            r=bounds)
    elif config.solver == "ga":
        return GeneticAlgorithm(seed=s,
                                num_params=config.n_params,
                                pop_size=20,
                                problem=problem,
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
                                lbs=[bounds[0] for _ in range(config.n_params)],
                                ubs=[bounds[1] for _ in range(config.n_params)])
    elif config.solver == "gpgo":
        return GPGO(seed=s,
                    num_params=config.n_params,
                    problem=problem,
                    r=bounds)
    elif config.solver == "bo":
        return BO(seed=s,
                  num_params=config.n_params,
                  problem=problem,
                  r=bounds)
    elif config.solver == "tpe":
        return TPE(seed=s,
                   num_params=config.n_params,
                   problem=problem,
                   r=bounds)
    raise ValueError("Invalid solver name: {}".format(config.solver))
