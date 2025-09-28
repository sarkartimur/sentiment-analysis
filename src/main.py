import pandas as pd
import numpy as np
# from model.bert_model_container import BERTClassifier
import data_loader as dl
from bert_container import BERTContainer
from constants import RANDOM_SEED
from model.bert_model_container import BERTClassifier, BERTModelContainer
from model.sklearn_model_container import SklearnModelContainer
import util


# Fix for non-deterministic cv/test accuracy
np.random.seed(RANDOM_SEED)

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)

# model = SklearnModelContainer(loader=dl.DataLoader(), model=util.svc_cv(), bert=BERTContainer())
# model.train()
# model.test()

MODEL_PATH = "F:\\IdeaProjects\\pretrained\\bert_imdb_sentiment_reduced_5l"
model = BERTModelContainer(loader=dl.DataLoader(), model=BERTClassifier(MODEL_PATH))
model.test()