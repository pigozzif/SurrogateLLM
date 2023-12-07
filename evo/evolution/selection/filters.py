import abc
import math

import numpy as np

from evo.utils.utilities import parse_largest_component


class Filter(object):

    @abc.abstractmethod
    def __call__(self, item):
        pass

    @classmethod
    def create_filter(cls, name: str):
        if name is None:
            return NoneFilter()
        elif name == "connected":
            return ConnectedFilter()
        elif name == "nonnegative":
            return NonNegativeFilter()
        raise ValueError("Invalid filter: {}".format(name))


class NoneFilter(Filter):

    def __call__(self, item):
        return True


class ConnectedFilter(Filter):

    def __call__(self, item):
        h, w = int(math.sqrt(item.size)), int(math.sqrt(item.size))
        a = np.where(item <= 0.0, 0, 1).reshape((h, w))
        largest_component_mask = parse_largest_component(item=a)
        return largest_component_mask is not None


class NonNegativeFilter(Filter):

    def __call__(self, item):
        return np.min(item) >= 0
