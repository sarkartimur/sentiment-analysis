from sklearn.linear_model import LinearRegression


def linear_regression(x, y):
    return LinearRegression(fit_intercept=True, copy_X=False, n_jobs=None).fit(x, y)
