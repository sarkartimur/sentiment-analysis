import numpy as np
import scipy.stats


def squared_error_total(values: np.ndarray):
    mean = values.mean()
    return ((values - mean)**2).sum()


def squared_error_regression(x: np.ndarray, y: np.ndarray, slope, intercept):
    return ((y - (slope*x + intercept))**2).sum()


def f_test(var_ratio, n_dof, d_dof):
    return 1 - scipy.stats.f.cdf(var_ratio, n_dof, d_dof)
