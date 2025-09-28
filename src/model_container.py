import numpy as np
from typing import Optional, Tuple, List
from protocols import Classifier, TrainTestSplit
from data_loader import DataLoader
from bert_container import BERTContainer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import util
import metrics
import time


class Settings:
    def __init__(
        self,
        do_reduce: bool=True,
        do_output_metrics: bool=True,
        do_analyze_errors: bool=True,
        bert_pooling: str='mean',
        num_dimensions: int=150,
        threshold: float=0.5
    ):
        self.do_reduce = do_reduce
        self.do_output_metrics = do_output_metrics
        self.do_analyze_errors = do_analyze_errors
        self.bert_pooling = bert_pooling
        self.num_dimensions = num_dimensions
        self.threshold = threshold



class ModelContainer:
    data: Optional[TrainTestSplit] = None
    reducer: Optional[PCA] = None

    def __init__(self, loader: DataLoader, model: Classifier, bert: BERTContainer, settings: Settings):
        self.loader = loader
        self.model = model
        self.bert = bert
        self.settings = settings


    def train(self) -> Optional[Tuple]:
        self.data = self.loader.load_data()
        
        train_embeddings = self.__extract_embeddings(self.data.X_train.to_list())
        test_embeddings = self.__extract_embeddings(self.data.X_test.to_list())
        
        if self.settings.do_reduce:
            train_embeddings, reducer = self.__reduce_dimensions(train_embeddings)
            self.reducer = reducer
            test_embeddings = reducer.transform(test_embeddings)

        print(f"\nTrain set shape: {train_embeddings.shape}")
        print(f"Test set shape: {test_embeddings.shape}\n")

        self.model.fit(train_embeddings, self.data.y_train)
        if isinstance(self.model, (GridSearchCV, RandomizedSearchCV)):
            print("Best parameters:", self.model.best_params_)
            print("Best cross-validation score:", self.model.best_score_)
            self.model = self.model.best_estimator_
    
        
        y_pred_proba = self.model.predict_proba(test_embeddings)[:, 1]
        y_pred = (y_pred_proba >= self.settings.threshold).astype(int)

        if self.settings.do_output_metrics:
            metrics.compute_metrics(self.data.y_test, y_pred, y_pred_proba)
            metrics.plot_graphs(self.data.y_test, y_pred, y_pred_proba)

        if self.settings.do_analyze_errors:
            return self.__analyze_errors(y_pred, y_pred_proba)

    def predict_list(self, X: List[str]) -> np.ndarray:
        embeddings = self.__extract_embeddings(X)
        if self.settings.do_reduce:
            embeddings = self.reducer.transform(embeddings)
        return self.model.predict_proba(embeddings)

    def predict(self, text: str):
        return self.predict_list([text])


    def __extract_embeddings(self, X: List[str]) -> np.ndarray[float]:
        print("\nExtracting BERT embeddings...")
        start_time = time.time()
        embeddings = self.bert.get_bert_embeddings(texts=X, pooling_strategy=self.settings.bert_pooling)
        print(f"Embedding extraction time: {time.time() - start_time:.2f} seconds")
        return embeddings

    def __reduce_dimensions(self, data: np.ndarray) -> Tuple[np.ndarray[float], PCA]:
        reducer = PCA(n_components=self.settings.num_dimensions)
        print(f"\nFitting PCA reducer")
        reduced = reducer.fit_transform(data)
        return reduced, reducer
    
    def __analyze_errors(self, y_pred, y_pred_proba):
        confident_false_positives = self.data.X_test[(self.data.y_test == 0) & (y_pred == 1) & (y_pred_proba > 0.9)]
        confident_false_negatives = self.data.X_test[(self.data.y_test == 1) & (y_pred == 0) & (y_pred_proba < 0.1)]
        
        print("\n")
        for i, fn in confident_false_positives[:10].items():
            print(f"Confident false positive at index {i}: {fn}")

        print("\n")
        for i, fn in confident_false_negatives[:10].items():
            print(f"Confident false negative at index {i}: {fn}")
        
        return confident_false_positives, confident_false_negatives
