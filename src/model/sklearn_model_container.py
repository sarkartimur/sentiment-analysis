import numpy as np
from typing import Optional, Tuple, List
from model.protocols import Classifier, ModelContainer, ModelSettings, TrainTestSplit
from data_loader import DataLoader
from bert_container import BERTContainer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import metrics
import time


class SklearnModelContainer(ModelContainer):
    reducer: Optional[PCA] = None

    def __init__(self, loader: DataLoader, model: Classifier, bert: BERTContainer, settings = ModelSettings()):
        self.loader = loader
        self.model = model
        self.bert = bert
        self.settings = settings


    def train(self) -> None:
        self.data = self.loader.load_data()
        
        train_embeddings = self.__extract_embeddings(self.data.X_train.to_list())
        
        if self.settings.do_reduce:
            train_embeddings, reducer = self.__reduce_dimensions(train_embeddings)
            self.reducer = reducer

        print(f"\nTrain set shape: {train_embeddings.shape}")

        self.model.fit(train_embeddings, self.data.y_train)
        if isinstance(self.model, (GridSearchCV, RandomizedSearchCV)):
            print("Best parameters:", self.model.best_params_)
            print("Best cross-validation score:", self.model.best_score_)
            self.model = self.model.best_estimator_

    def test(self) -> None:
        if (self.data is None):
            self.data = self.loader.load_data()
        
        test_embeddings = self.__extract_embeddings(self.data.X_test.to_list())
        
        # todo consider case when model loaded from file (train not called before test)
        if self.settings.do_reduce:
            test_embeddings = self.reducer.transform(test_embeddings)

        print(f"Test set shape: {test_embeddings.shape}\n")

        y_pred_proba = self.model.predict_proba(test_embeddings)[:, 1]
        y_pred = (y_pred_proba >= self.settings.threshold).astype(int)

        self._output_metrics(y_pred, y_pred_proba)


    def predict_list(self, X: List[str]) -> np.ndarray:
        embeddings = self.__extract_embeddings(X)
        if self.settings.do_reduce:
            embeddings = self.reducer.transform(embeddings)
        return self.model.predict_proba(embeddings)


    def __extract_embeddings(self, X: List[str]) -> np.ndarray:
        print("\nExtracting BERT embeddings...")
        start_time = time.time()
        embeddings = self.bert.get_bert_embeddings(texts=X, pooling_strategy=self.settings.bert_pooling)
        print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")
        return embeddings

    def __reduce_dimensions(self, data: np.ndarray) -> Tuple[np.ndarray, PCA]:
        reducer = PCA(n_components=self.settings.num_dimensions)
        print(f"\nFitting PCA reducer")
        reduced = reducer.fit_transform(data)
        return reduced, reducer
