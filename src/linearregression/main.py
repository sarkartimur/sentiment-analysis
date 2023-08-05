from matplotlib import pyplot as plt
import numpy as np
from sklearn_lr import linear_regression
from singular_lr_scratch import fit

x = np.array([5, 15, 25, 35, 45, 55])
x2 = x.reshape(-1, 1)
y = np.array([30, 21, 27, 39, 40, 37])
m = 0
b = 0
lr = 0.0001
epochs = 10000

m, b = fit(epochs, x, y, m, b, lr)
model = linear_regression(x2, y)

fig, ax = plt.subplots()
ax.scatter(x, y, color='g')
ax.plot([m*x + b for x in range(x[0], x[-1])], color='r')
ax.plot(x2, model.predict(x2), color='b')
plt.show()