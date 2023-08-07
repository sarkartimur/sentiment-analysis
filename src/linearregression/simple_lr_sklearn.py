import logging
from numpy import ndarray
from sklearn.linear_model import LinearRegression


logger = logging.getLogger(__name__)


def linear_regression(x: ndarray, y: ndarray) -> LinearRegression:
    # Note: sklearn uses the least squares method for optimization (solve Ax=b)
    model = LinearRegression(fit_intercept=True, copy_X=False, n_jobs=None).fit(x, y)
    logger.info(f'Slope: {model.coef_}, Intercept: {model.intercept_}')
    return model
