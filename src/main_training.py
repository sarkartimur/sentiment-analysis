import pandas as pd
import numpy as np
from data_loader import DataLoader
from model.sklearn.bert_wrapper import BERTWrapper
from constants import RANDOM_SEED
from model.bert.bert_model_container import BERTClassifier, BERTModelContainer
from model.sklearn.sklearn_model_container import SklearnModelContainer
import util


# Fix for non-deterministic cv/test accuracy
np.random.seed(RANDOM_SEED)

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)

# model = SklearnModelContainer(loader=dl.DataLoader(), model=util.svc_cv(), bert=BERTWrapper())
# model.train()
# model.test()

model = BERTModelContainer(loader=DataLoader(), model=BERTClassifier(), calibration_method=None)
model.train()
model.test()