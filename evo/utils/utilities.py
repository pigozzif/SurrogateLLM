import math
import random

import numpy as np
from scipy import ndimage


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def weighted_random_by_dct(dct):
    rand_val = random.random()
    total = 0
    for k, v in dct.items():
        total += v
        if rand_val <= total:
            return k
    raise RuntimeError("Could not sample from dictionary")


def dominates(ind1, ind2, attribute_name, maximize):
    """Returns 1 if ind1 dominates ind2 in a shared attribute, -1 if ind2 dominates ind1, 0 otherwise."""
    if ind1.fitness[attribute_name] > ind2.fitness[attribute_name]:
        ans = 1
    elif ind1.fitness[attribute_name] < ind2.fitness[attribute_name]:
        ans = -1
    else:
        ans = 0
    return ans if maximize else -ans


def parse_largest_component(solution):
    binary_matrix = np.where(solution <= 0.0, 0, 1)
    labeled_array, num_features = ndimage.label(binary_matrix)
    if num_features == 0:
        return None
    component_sizes = np.bincount(labeled_array.ravel())
    largest_component_label = np.argmax(component_sizes[1:]) + 1
    largest_component_mask = labeled_array == largest_component_label
    h, w = solution.shape
    if not (largest_component_mask[0, w // 2] and largest_component_mask[h - 1, round(w * 0.3)] and largest_component_mask[h - 1, round(w * 0.6)]):
        return None
    return largest_component_mask
