from locale import normalize
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression(fit_intercept=True, copy_X=False, n_jobs=None).fit(x, y)

fig, ax = plt.subplots()
ax.scatter(x, y,color='g')
ax.plot(x, model.predict(x),color='k')
fig.show()

print()