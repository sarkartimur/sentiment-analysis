from matplotlib import pyplot as plt
import numpy as np
from sklearn_lr import linear_regression
from singular_lr_scratch import fit


PREDICTOR = np.array([5, 15, 25, 35, 45, 55])
SK_PREDICTOR = PREDICTOR.reshape(-1, 1)
TARGET = np.array([30, 21, 27, 39, 40, 37])
LEARN_RATE = 0.0008
EPOCHS = 10000
slope = 0
intercept = 0

slope, intercept = fit(EPOCHS, slope, intercept, PREDICTOR, TARGET, LEARN_RATE)
model = linear_regression(SK_PREDICTOR, TARGET)

fig, ax = plt.subplots()
ax.scatter(PREDICTOR, TARGET, color='g')
ax.plot([slope*x + intercept for x in range(PREDICTOR[0], PREDICTOR[-1])], color='r')
ax.plot(SK_PREDICTOR, model.predict(SK_PREDICTOR), color='b')
plt.show()