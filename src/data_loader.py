import pandas as pd
from datasets import load_dataset
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from util import RANDOM_SEED


class DataLoader:

    __MINORITY_CLASS = 1

    def __init__(self, dataset_name='imdb'):
        self.__dataset_name = dataset_name

    def load_data_dict(self, sample_size=2000, test_ratio=0.25):
        X_train, y_train, X_test, y_test = self.load_data(sample_size, test_ratio)
        return DatasetDict({
            'train': Dataset.from_dict({'features': X_train, 'labels': y_train}),
            'test': Dataset.from_dict({'features': X_test, 'labels': y_test})
        })

    def load_data(self, sample_size=3000, test_ratio=0.4, calibration_ratio=None, imbalance_ratio=None):
        print(f"Loading {self.__dataset_name} dataset...")
        dataset = load_dataset(self.__dataset_name)
        df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)

        print(df.head())

        X = df['text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=int(sample_size * (1 - test_ratio)),
            test_size=int(sample_size * test_ratio),
            stratify=y,
            random_state=RANDOM_SEED
        )

        if calibration_ratio is not None:
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_test, y_test, test_size=calibration_ratio, random_state=RANDOM_SEED, stratify=y_test
            )

        print(f"\nTraining set - Positive: {sum(y_train)}, Negative: {len(y_train) - sum(y_train)}")
        print(f"Testing set - Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")
        if calibration_ratio is not None:
            print(f"Calibration set - Positive: {sum(y_cal)}, Negative: {len(y_cal) - sum(y_cal)}")

        if imbalance_ratio is not None:
            X_train, y_train = self.__add_imbalance(X_train, y_train, imbalance_ratio)

        print('\n')

        return (
            (X_train, y_train, X_test, y_test, X_cal, y_cal)
            if calibration_ratio is not None else
            (X_train, y_train, X_test, y_test)
        )

    def __add_imbalance(self, X_train, y_train, imbalance_ratio):
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})

        majority_df = train_df[train_df['label'] != self.__MINORITY_CLASS]
        minority_df = train_df[train_df['label'] == self.__MINORITY_CLASS]

        target_minority_count = int(len(majority_df) * imbalance_ratio)
        if len(minority_df) > target_minority_count:
            minority_df = minority_df.sample(n=target_minority_count, random_state=RANDOM_SEED)
        else:
            print(f"Warning: Cannot achieve {imbalance_ratio} ratio - not enough minority samples")

        train_df = pd.concat([majority_df, minority_df])
        # Shuffle
        train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        X_train = train_df['text']
        y_train = train_df['label']

        print(f"Imbalanced training set - Class {self.__MINORITY_CLASS} as minority")
        print(f"Positive: {sum(y_train)}, Negative: {len(y_train) - sum(y_train)}")
        print(f"Imbalance ratio: ~1:{round(1/imbalance_ratio)}")

        return X_train, y_train
