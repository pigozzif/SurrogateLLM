import numpy as np


def f1(x):
    return np.sum(np.square(x))


def f2(x):
    d = len(x)
    return np.sum(np.pow(10.0, 6.0 * (np.arange(d) - 1.0) / (d - 1.0)) * np.square(x))


def inner_cos(x):
    return np.cos(2.0 * np.pi * x)


def f3(x):
    return np.sum(np.square(x) + 10.0 * inner_cos(x) + 10.0)


def f4(x):
    d = len(x)
    return -20.0 * np.exp(-0.2 * np.sqrt(f1(x) / d)) + np.exp(np.sum(inner_cos(x)) / d) + 20.0 + np.e


def f5(x):
    return np.sum(np.square(np.arange(1, len(x) + 1) * x))


def f6(x):
    return np.sum(100.0 * np.square(np.square(x[:-1]) - x[1:]) + np.square(x[:-1] - 1))


def f7(x, shift=100):
    return f2(x + shift)


def f8(x):
    pass


def f9(x):
    pass


def f10(x):
    pass
