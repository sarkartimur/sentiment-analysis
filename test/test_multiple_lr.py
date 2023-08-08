from matplotlib import pyplot as plt
import numpy as np
np.random.seed(777)
from sklearn.linear_model import LinearRegression


def test_multiple_linear_regression():
    predictor = np.random.randint(0, 100, size=(10, 2))
    target = np.random.randint(0, 100, size=10)

    model = LinearRegression().fit(predictor, target)

    coef = model.coef_
    intercept = model.intercept_

    x, y = np.linspace(0, 100, 10), np.linspace(0, 100, 10)
    xs, ys = np.meshgrid(x, y)
    zs = xs*coef[0] + ys*coef[1] + intercept

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.view_init(azim=-60, elev=6)
    
    ax.scatter([i[0] for i in predictor], [i[1] for i in predictor], target, color='g')
    ax.plot_surface(xs, ys, zs, alpha=0.5, cmap='plasma')
    plt.show()