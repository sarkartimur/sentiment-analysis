import logging
from matplotlib import pyplot as plt
import numpy as np
from simple_lr_sklearn import linear_regression
from simple_lr_scratch import SimpleLinearRegression


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger('matplotlib').setLevel(logging.INFO)


PREDICTOR = np.array([5, 15, 25, 35, 45, 55])
SK_PREDICTOR = PREDICTOR.reshape(-1, 1)
TARGET = np.array([30, 21, 27, 39, 40, 37])
LEARN_RATE = 0.0001
EPOCHS = 25000

model = SimpleLinearRegression(PREDICTOR, TARGET, LEARN_RATE, EPOCHS).fit()
sklearn_model = linear_regression(SK_PREDICTOR, TARGET)
# Compute slope using the formula (for reference)
slope = np.corrcoef(PREDICTOR, TARGET)*TARGET.std()/PREDICTOR.std()
logger.info(f'Reference slope: {slope}')

fig, ax = plt.subplots()
ax.scatter(PREDICTOR, TARGET, color='g')
ax.plot(model.plot_line(), color='r')
ax.plot(SK_PREDICTOR, sklearn_model.predict(SK_PREDICTOR), color='b')
plt.show()
