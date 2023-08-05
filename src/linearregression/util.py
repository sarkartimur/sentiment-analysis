import numpy as np
import scipy.stats
  

def squared_error_mean(values: np.ndarray):
    mean = values.mean()
    return ((values - mean)**2).sum()

def squared_error_line(x: np.ndarray, y: np.ndarray, slope, intercept):
    return ((y - (slope*x + intercept))**2).sum()


def f_test(var_ratio, n_dof, d_dof):
    p_value = 1 - scipy.stats.f.cdf(var_ratio, n_dof, d_dof)
    return p_value