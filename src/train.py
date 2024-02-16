import os

from catboost import CatBoostClassifier
from keras.layers import Dense, Input
from keras.models import Sequential
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier

from automl.core import auto_ml
from automl.launch_strategy import simple_strategy
from automl.optimize_strategy import optimize_strategy
from automl.tf_wrap import AutoTFClassifier
from src.components.data.process_data import process_data
from src.components.data.read_dataset import read_dataset


def train():
    dataset = read_dataset("time_series_60min_singleindex.csv")
    x_data, y_data = process_data(dataset)

    arch = [
        (
            AutoTFClassifier,
            {
                "name": "AutoTF",
                "init_args": {
                    "network": Sequential(
                        [
                            Input(shape=x_data.shape[1]),
                            Dense(64, activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(4, activation="softmax"),
                        ]
                    ),
                    "n_epochs": 20,
                },
                "hyper_parameters": {
                    "network": [
                        Sequential(
                            [
                                Input(shape=x_data.shape[1]),
                                Dense(64, activation="relu"),
                                Dense(32, activation="relu"),
                                Dense(4, activation="softmax"),
                            ]
                        )
                    ],
                    "n_epochs": [10, 20, 30],
                    "batch_size": [32, 64, 128],
                },
            },
        ),
        (
            KNeighborsClassifier,
            {
                "name": "Knn",
                "hyper_parameters": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
            },
        ),
        # (
        #     RandomForestClassifier,
        #     {
        #         "name": "RandomForest",
        #         "hyper_parameters": {
        #             "criterion": ["gini", "entropy"],
        #             "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #             "min_samples_split": [10, 20, 30, 40, 50],
        #             "max_depth": [4, 6, 8, 10, 12],
        #         },
        #     },
        # ),
        # (
        #     CatBoostClassifier,
        #     {
        #         "hyper_parameters": {
        #             "learning_rate": [0.05, 0.1, 0.2],
        #             "depth": [2, 3, 4, 5, 6],
        #             "rsm": [0.7, 0.8, 0.9, 1],
        #             "subsample": [0.7, 0.8, 0.9, 1],
        #             "min_data_in_leaf": [1, 5, 10, 15, 20, 30, 50],
        #             "bootstrap_type": ["Bernoulli"],
        #             "eval_metric": ["MultiClass"],
        #             "verbose": [False],
        #         },
        #     },
        # ),
        # (
        #     LGBMClassifier,
        #     {
        #         "hyper_parameters": {
        #             "num_leaves": [3, 7, 15, 31],
        #             "learning_rate": [0.05, 0.075, 0.1, 0.15],
        #             "feature_fraction": [0.8, 0.9, 1.0],
        #             "bagging_fraction": [0.8, 0.9, 1.0],
        #             "min_data_in_leaf": [5, 10, 15, 20, 30, 50],
        #             "verbose": [-100],
        #         }
        #     },
        # ),
    ]

    metrics = [
        (accuracy_score, {"name": "accuracy"}),
        (f1_score, {"name": "f1", "args": {"average": "weighted"}}),
    ]

    list_of_strategies = [
        simple_strategy,
        optimize_strategy,
    ]

    print("\n\n ############### Start AutoML ############### \n\n")
    result = auto_ml(x_data, y_data, arch, metrics, list_of_strategies)


if __name__ == "__main__":
    train()
