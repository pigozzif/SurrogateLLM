import abc
import random

import numpy as np

from evo.evolution.selection.filters import Filter


class GenotypeFactory(object):

    def __init__(self, genotype_filter: Filter):
        self.genotype_filter = genotype_filter

    def create_population(self, pop_size: int) -> list:
        pop = []
        while len(pop) < pop_size:
            new_born = self.create()
            if self.genotype_filter(new_born):
                pop.append(new_born)
        return pop

    @abc.abstractmethod
    def create(self):
        pass

    @classmethod
    def create_factory(cls, name: str, genotype_filter: Filter, **kwargs):
        if name == "uniform_float":
            return UniformFloatFactory(genotype_filter=genotype_filter,
                                       n=kwargs["n"],
                                       lbs=kwargs["lbs"],
                                       ubs=kwargs["ubs"])
        raise ValueError("Invalid genotype factory name: {}".format(name))


class UniformFloatFactory(GenotypeFactory):

    def __init__(self, genotype_filter, n: int, lbs, ubs):
        super().__init__(genotype_filter)
        self.n = n
        self.lbs = lbs
        self.ubs = ubs

    def create(self) -> np.ndarray:
        return np.array([random.random() * (self.ubs[i] - self.lbs[i]) + self.lbs[i] for i in range(self.n)])
