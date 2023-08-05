from matplotlib import pyplot as plt
import numpy as np
from sklearn_lr import linear_regression
from singular_lr_scratch import fit


INPUT = np.array([5, 15, 25, 35, 45, 55])
SK_INPUT = INPUT.reshape(-1, 1)
OUTPUT = np.array([30, 21, 27, 39, 40, 37])
LEARN_RATE = 0.0008
EPOCHS = 10000
slope = 0
intercept = 0

slope, intercept = fit(EPOCHS, slope, intercept, INPUT, OUTPUT, LEARN_RATE)
model = linear_regression(SK_INPUT, OUTPUT)

fig, ax = plt.subplots()
ax.scatter(INPUT, OUTPUT, color='g')
ax.plot([slope*x + intercept for x in range(INPUT[0], INPUT[-1])], color='r')
ax.plot(SK_INPUT, model.predict(SK_INPUT), color='b')
plt.show()