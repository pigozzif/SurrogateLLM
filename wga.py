from expensiveoptimbenchmark.solvers.utils import Monitor
from evo.evolution.algorithms import GeneticAlgorithm
from evo.evolution.objectives import ObjectiveDict


def optimize_GA(problem, max_evals, rand_evals, seed, log=None):
    monitor = Monitor("ga", problem=problem, log=log)
    objectives_dict = ObjectiveDict()
    objectives_dict.add_objective(name="fitness", maximize=False, best_value=0.0, worst_value=float("inf"))
    ga = GeneticAlgorithm(seed=seed,
                          num_params=problem.dims(),
                          pop_size=rand_evals,
                          genotype_factory="uniform_float",
                          objectives_dict=objectives_dict,
                          survival_selector="worst",
                          parent_selector="tournament",
                          offspring_size=rand_evals // 2,
                          overlapping=True,
                          remap=False,
                          genetic_operators={"gaussian_mut": 1.0},
                          genotype_filter=None,
                          tournament_size=5,
                          mu=0.0,
                          sigma=0.01,
                          n=problem.dims(),
                          lbs=problem.lbs(),
                          ubs=problem.ubs())

    def f(x):
        monitor.commit_start_eval()
        r = problem.evaluate(x)
        monitor.commit_end_eval(x, r)
        return r

    monitor.start()
    evaluated = 0
    while evaluated < max_evals:
        solutions = ga.ask()
        fitness_list = [f(x) for x in solutions]
        ga.tell(fitness_list)
        evaluated += len(solutions)
    monitor.end()
    return monitor.best_x, monitor.best_fitness, monitor
