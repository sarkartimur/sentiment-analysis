from numpy import ndarray
from sklearn.linear_model import LinearRegression


def linear_regression(x: ndarray, y: ndarray) -> LinearRegression:
    model = LinearRegression(fit_intercept=True, copy_X=False, n_jobs=None).fit(x, y)
    print(f'Slope: {model.coef_}, Intercept: {model.intercept_}')
    return model
