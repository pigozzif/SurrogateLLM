import abc
import random
from abc import ABC
from typing import Dict

import numpy as np

from .objectives import ObjectiveDict
from .operators.operator import GeneticOperator
from .selection.filters import Filter
from .selection.selector import Selector
from ..representations.factory import GenotypeFactory
from ..representations.population import Population, Individual, Comparator
from ..utils.utilities import weighted_random_by_dct


class StochasticSolver(abc.ABC):

    def __init__(self, seed, num_params, pop_size):
        self.seed = seed
        self.num_params = num_params
        self.pop_size = pop_size

    @abc.abstractmethod
    def ask(self):
        pass

    @abc.abstractmethod
    def tell(self, fitness_list):
        pass

    @abc.abstractmethod
    def result(self):
        pass

    @abc.abstractmethod
    def get_num_evaluated(self):
        return


class GPGO(StochasticSolver):

    def __init__(self, seed, num_params, f, r, init_evals=3):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=1)

        from pyGPGO.covfunc import matern32
        from pyGPGO.acquisition import Acquisition
        from pyGPGO.surrogates.GaussianProcess import GaussianProcess
        from pyGPGO.GPGO import GPGO
        cov = matern32()
        gp = GaussianProcess(cov)
        acq = Acquisition(mode="UCB")
        param = {"x{}".format(i): ("cont", list(r)) for i in range(num_params)}
        self.gpgo = GPGO(gp, acq, f, param)
        self.it = 0
        self.init_evals = init_evals
        self.gpgo.init_evals = init_evals

    def _firstRun(self):
        self.gpgo.X = np.empty((self.init_evals, len(self.gpgo.parameter_key)))
        self.gpgo.y = np.empty((self.init_evals,))
        for i in range(self.init_evals):
            s_param = self.gpgo._sampleParam()
            s_param_val = list(s_param.values())
            self.gpgo.X[i] = s_param_val

    def _updateGP(self, f_new):
        self.gpgo.GP.update(np.atleast_2d(self.gpgo.best), np.atleast_1d(f_new))
        self.gpgo.tau = np.max(self.gpgo.GP.y)
        self.gpgo.history.append(self.gpgo.tau)

    def ask(self):
        if self.it == 0:
            self._firstRun()
            return [self.gpgo.X[i] for i in range(self.init_evals)]
        self.gpgo._optimizeAcq()
        return [self.gpgo.best]

    def tell(self, fitness_list):
        if self.it == 0:
            for i in range(self.init_evals):
                self.gpgo.y[i] = - fitness_list[i]
            self.gpgo.GP.fit(self.gpgo.X, self.gpgo.y)
            self.gpgo.tau = np.max(self.gpgo.y)
            self.gpgo.history.append(self.gpgo.tau)
            self.it += 1
            return
        self.gpgo._optimizeAcq()
        self._updateGP(f_new=-fitness_list[0])
        self.it += 1

    def result(self):
        best_genotype, best_fitness = self.gpgo.getResult()
        return [v for v in best_genotype.values()], - best_fitness

    def get_num_evaluated(self):
        return len(self.gpgo.history) + self.init_evals - 1


class PopulationBasedSolver(StochasticSolver, ABC):

    def __init__(self, seed: int,
                 num_params: int,
                 pop_size: int,
                 genotype_factory: str,
                 objectives_dict: ObjectiveDict,
                 remap: bool,
                 genetic_operators: Dict[str, float],
                 comparator: str,
                 genotype_filter: str = None,
                 **kwargs):
        super().__init__(seed=seed, num_params=num_params, pop_size=pop_size)
        self.pop_size = pop_size
        self.remap = remap
        self.pop = Population(pop_size=pop_size,
                              genotype_factory=GenotypeFactory.create_factory(name=genotype_factory,
                                                                              genotype_filter=Filter.create_filter(
                                                                                  genotype_filter), **kwargs),
                              objectives_dict=objectives_dict,
                              comparator=comparator)
        self.genetic_operators = {GeneticOperator.create_genetic_operator(name=k,
                                                                          genotype_filter=Filter.create_filter(
                                                                              genotype_filter), **kwargs):
                                      v for k, v in genetic_operators.items()}

    def get_best(self) -> Individual:
        return self.pop.get_best()


class RandomSearch(PopulationBasedSolver):

    def __init__(self, seed, num_params, objectives_dict, r):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=1,
                         genotype_factory="uniform_float",
                         objectives_dict=objectives_dict,
                         remap=False,
                         genetic_operators={},
                         genotype_filter="none",
                         comparator="lexicase",
                         range=r,
                         n=num_params)
        self.best_fitness = float("inf")
        self.best_genotype = None

    def ask(self):
        if self.pop.gen == 0:
            self.pop.clear()
        for g in self.pop.init_random_individuals(n=self.pop_size):
            self.pop.add_individual(g)
        return [ind.genotype for ind in self.pop]

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = {"fitness": f}
            ind.evaluated = True
        best_idx = np.argmin(fitness_list)
        if self.best_fitness <= fitness_list[best_idx]:
            self.best_fitness = fitness_list[best_idx]
            self.best_genotype = self.pop[best_idx]
        self.pop.clear()

    def result(self):
        return self.best_genotype, self.best_fitness


class GeneticAlgorithm(PopulationBasedSolver):

    def __init__(self, seed, num_params, pop_size, genotype_factory, objectives_dict, survival_selector: str,
                 parent_selector: str, offspring_size: int, overlapping: bool, remap, genetic_operators,
                 genotype_filter, **kwargs):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=pop_size,
                         genotype_factory=genotype_factory,
                         objectives_dict=objectives_dict,
                         remap=remap,
                         genetic_operators=genetic_operators,
                         genotype_filter=genotype_filter,
                         comparator="lexicase",
                         **kwargs)
        self.survival_selector = Selector.create_selector(name=survival_selector, **kwargs)
        self.parent_selector = Selector.create_selector(name=parent_selector, **kwargs)
        self.offspring_size = offspring_size
        self.overlapping = overlapping
        self.num_evaluated = 0

    def _build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def _trim_population(self) -> None:
        while len(self.pop) > self.pop_size:
            self.pop.remove_individual(self.survival_selector.select(population=self.pop, n=1)[0])

    def ask(self):
        if self.pop.gen != 0:
            for child_genotype in self._build_offspring():
                self.pop.add_individual(genotype=child_genotype)
        return [ind.genotype for ind in self.pop if not ind.evaluated]

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = {"fitness": f}
            ind.evaluated = True
            self.num_evaluated += 1
        if self.pop.gen != 0:
            self._trim_population()
        self.pop.gen += 1

    def result(self):
        best = self.get_best()
        return best.genotype, best.fitness["fitness"]

    def get_num_evaluated(self):
        return self.num_evaluated


class AFPO(GeneticAlgorithm):

    def __init__(self, seed, num_params, pop_size, genotype_factory, objectives_dict, offspring_size: int,
                 remap, genetic_operators, genotype_filter, **kwargs):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=pop_size,
                         genotype_factory=genotype_factory,
                         objectives_dict=objectives_dict,
                         remap=remap,
                         genetic_operators=genetic_operators,
                         survival_selector="tournament_pareto",
                         parent_selector="tournament",
                         offspring_size=offspring_size - 1,
                         overlapping=True,
                         genotype_filter=genotype_filter,
                         **kwargs)
        self.pareto_comparator = Comparator.create_comparator(name="pareto", objective_dict=self.pop.objectives_dict)

    def _select_individual(self, population: Population) -> Individual:
        ind1, ind2 = tuple(population.sample(n=2))
        c = self.pareto_comparator.compare(ind1=ind1, ind2=ind2)
        if c == -1:
            return ind1
        elif c == 1:
            return ind2
        return random.choice([ind1, ind2])

    def _build_offspring(self) -> list:
        children_genotypes = super()._build_offspring()
        children_genotypes.extend(self.pop.init_random_individuals(n=1))
        return children_genotypes

    def _trim_population(self) -> None:
        while len(self.pop) > self.pop_size:
            self.pop.remove_individual(self._select_individual(population=self.pop))

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = {"fitness": f, "age": 0}
            ind.evaluated = True
        if self.pop.gen != 0:
            self._trim_population()
        self.pop.gen += 1
        self.pop.update_ages()


class CMAES(StochasticSolver):

    def __init__(self, seed, num_params, pop_size, sigma_init):
        super().__init__(seed, num_params, pop_size)
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.solutions = None
        self.it = 0

        import cma
        self.es = cma.CMAEvolutionStrategy(self.num_params * [0], self.sigma_init, {"popsize": self.pop_size})

    def ask(self):
        self.solutions = self.es.ask()
        self.it += 1
        return self.solutions

    def tell(self, fitness_list):
        self.es.tell(self.solutions, [-f for f in fitness_list])

    def result(self):
        r = self.es.result
        return r[0], r[1]

    def get_num_evaluated(self):
        return self.it * self.pop_size


class NSGAII(PopulationBasedSolver):

    def __init__(self, seed, pop_size, genotype_factory, offspring_size: int, remap, genetic_operators,
                 genotype_filter, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size,
                         genotype_factory=genotype_factory,
                         remap=remap,
                         genetic_operators=genetic_operators,
                         genotype_filter=genotype_filter,
                         comparator="pareto", **kwargs)
        self.offspring_size = offspring_size
        self.fronts = {}
        self.dominates = {}
        self.dominated_by = {}
        self.crowding_distances = {}
        self.parent_selector = Selector.create_selector(name="tournament_crowded",
                                                        crowding_distances=self.crowding_distances, fronts=self.fronts,
                                                        **kwargs)
        self.best_sensing = None
        self.best_locomotion = None

    def _fast_non_dominated_sort(self) -> None:
        self.fronts.clear()
        self.dominates.clear()
        self.dominated_by.clear()
        for p in self.pop:
            self.dominated_by[p.id] = 0
            for q in self.pop:
                if p.id == q.id:
                    continue
                elif p > q:
                    if p.id not in self.dominates:
                        self.dominates[p.id] = [q]
                    else:
                        self.dominates[p.id].append(q)
                elif p < q:
                    self.dominated_by[p.id] += 1
            if self.dominated_by[p.id] == 0:
                if 0 not in self.fronts:
                    self.fronts[0] = [p]
                else:
                    self.fronts[0].append(p)
        if not self.fronts:
            self.fronts[0] = [ind for ind in self.pop]
            return
        i = 0
        while len(self.fronts[i]):
            self.fronts[i + 1] = []
            for p in self.fronts[i]:
                for q in self.dominates.get(p.id, []):
                    self.dominated_by[q.id] -= 1
                    if self.dominated_by[q.id] == 0:
                        self.fronts[i + 1].append(q)
            i += 1
        self.fronts.pop(i)
        self.crowding_distances.clear()
        for front in self.fronts.values():
            self._crowding_distance_assignment(individuals=front)

    def _crowding_distance_assignment(self, individuals: list) -> None:
        for individual in individuals:
            self.crowding_distances[individual.id] = 0.0
        for rank, goal in self.pop.objectives_dict.items():
            individuals.sort(key=lambda x: x.fitness[goal["name"]], reverse=goal["maximize"])
            self.crowding_distances[individuals[0].id] = float("inf")
            self.crowding_distances[individuals[len(individuals) - 1].id] = float("inf")
            for i in range(1, len(individuals) - 1):
                self.crowding_distances[individuals[i].id] += (individuals[i + 1].fitness[goal["name"]] -
                                                               individuals[i - 1].fitness[goal["name"]]) / \
                                                              (abs(goal["best_value"] - goal["worst_value"]))

    def _build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def _trim_population(self) -> None:
        self._fast_non_dominated_sort()
        i = 0
        n = 0
        while n + len(self.fronts[i]) <= self.pop_size:
            n += len(self.fronts[i])
            i += 1
        self.fronts[i].sort(key=lambda x: self.crowding_distances[x.id])
        for j in range(len(self.fronts[i]) - self.pop_size + n):
            self.pop.remove_individual(ind=self.fronts[i][j])
        i += 1
        while i in self.fronts:
            for ind in self.fronts[i]:
                self.pop.remove_individual(ind=ind)
            i += 1

    def ask(self):
        if self.pop.gen != 0:
            for child_genotype in self._build_offspring():
                self.pop.add_individual(genotype=child_genotype)
        else:
            self._fast_non_dominated_sort()
        return [ind.genotype for ind in self.pop]

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = f
            ind.evaluated = True
        if self.pop.gen != 0:
            self._trim_population()
        self.pop.gen += 1

    def result(self):
        return [best.genotype for best in self.fronts[0]], [best.fitness["fitness"] for best in self.fronts[0]]
