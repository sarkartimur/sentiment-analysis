import pandas as pd
import numpy as np
import data_loader as dl
from bert_container import BERTContainer
from constants import RANDOM_SEED
from model_container import ModelContainer, Settings
import util


# Fix for non-deterministic cv/test accuracy
np.random.seed(RANDOM_SEED)

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)

model = ModelContainer(loader=dl.DataLoader(), model=util.svc(), bert=BERTContainer(), settings=Settings())

model.train()