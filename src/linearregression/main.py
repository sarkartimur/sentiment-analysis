from matplotlib import pyplot as plt
import numpy as np
from simple_lr_sklearn import linear_regression
from simple_lr_scratch import SimpleLinearRegression


PREDICTOR = np.array([5, 15, 25, 35, 45, 55])
SK_PREDICTOR = PREDICTOR.reshape(-1, 1)
TARGET = np.array([30, 21, 27, 39, 40, 37])
LEARN_RATE = 0.0008
EPOCHS = 10000

model = SimpleLinearRegression(PREDICTOR, TARGET, LEARN_RATE, EPOCHS).fit()
sklearn_model = linear_regression(SK_PREDICTOR, TARGET)

fig, ax = plt.subplots()
ax.scatter(PREDICTOR, TARGET, color='g')
ax.plot(model.plot_line(), color='r')
ax.plot(SK_PREDICTOR, sklearn_model.predict(SK_PREDICTOR), color='b')
plt.show()
