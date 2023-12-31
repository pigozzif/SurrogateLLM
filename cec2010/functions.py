import numpy as np


def sphere(x):
    return np.sum(np.square(x))


def elliptic(x):
    d = len(x)
    return np.sum(np.power(10.0, 6.0 * (np.arange(d) - 1.0) / (d - 1.0)) * np.square(x))


def inner_cos(x):
    return np.cos(2.0 * np.pi * x)


def rastrigin(x):
    return np.sum(np.square(x) + 10.0 * inner_cos(x) + 10.0)


def ackley(x):
    d = len(x)
    return -20.0 * np.exp(-0.2 * np.sqrt(sphere(x) / d)) + np.exp(np.sum(inner_cos(x)) / d) + 20.0 + np.e


def schwefel(x):
    return np.sum(np.square(np.arange(1, len(x) + 1) * x))


def rosenbrock(x):
    return np.sum(100.0 * np.square(np.square(x[:-1]) - x[1:]) + np.square(x[:-1] - 1))
