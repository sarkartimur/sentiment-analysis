from dataclasses import replace
import pandas as pd
import numpy as np
import shap
from model.data_loader import DataLoader
from model.bert.bert_classifier import BertClassifierSettings
from model.protocols import ModelSettings
from model.shap_explain import BERTExplainer
from model.sklearn.bert_container import BERTContainer
from model.constants import BERT_MODEL, RANDOM_SEED
from model.bert.bert_model_agent import BERTClassifier, BERTModelContainer
from model.sklearn.sklearn_model_agent import SklearnModelAgent
import model.util as util


# Fix for non-deterministic cv/test accuracy
np.random.seed(RANDOM_SEED)

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)

# model = SklearnModelContainer(loader=DataLoader(), model=util.xgboost(), bert=BERTWrapper())
# model.train()
# model.test()

bert_settings = BertClassifierSettings(local_model=True, temperature_scale = 1.5)
cls = BERTClassifier(settings=bert_settings)

model = BERTModelContainer(loader=DataLoader(), model=cls, calibration_method=None)
# model.train()
# cls.save()
# model.test()

text = "This movie deserved better. Mike Judge's satirical wit brought to light something too many in this country are trying to deny... we're getting dumber as a society.<br /><br />Could the 24-hour-a-day Anna Nicole coverage be any more proof? Mike Judge paints a frightening future, where the dumb survive and thrive. Makes you stop and think, and laugh. Can you look at the world and not ask are we getting dumber? Are we being overtaking by the human trash as well as our own trash? (Beware of landslides).<br /><br />The movie is really funny. I'd tell you more of the plot but I don't want to spoil it.<br /><br />So why release this film with ZERO promotion? Could it be that the stupid are already taking over?"
exp = BERTExplainer(bert_wrapper=cls, predict_method=model.predict_multiple)
shap_exp = exp.explain_prediction(text)
shap.plots.waterfall(shap_exp[0, :, 0], max_display=100)