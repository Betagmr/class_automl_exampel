from random import choice

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from automl.core import calculate_metrics


def simple_strategy(
    x_data,
    y_data,
    arch_list,
    metrics,
    previous_info=None,
):
    kfs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split_indexes = list(kfs.split(x_data, y_data))

    models_results = []
    for model_init, args in arch_list:
        model_name = args.get("name", model_init.__name__)
        print("Running on model:", model_name)

        y_test_list = []
        y_predict_list = []
        for train_idx, test_idx in split_indexes:
            # Split the data
            x_train, x_test = x_data[train_idx], x_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]

            # Train the model
            init_args = get_init_args(args)
            model = model_init(**init_args)
            model.fit(x_train, y_train)

            # Make predictions
            predictions = model.predict(x_test, **args.get("predict_args", {}))
            y_predict_list.append(args.get("y_post", lambda y: y)(predictions))
            y_test_list.append(y_test)

        model_metrics = calculate_metrics(
            np.concatenate(y_test_list),
            np.concatenate(y_predict_list),
            metrics,
        )

        model_info = {"model": model_name} | model_metrics
        print(model_info, "\n")

        # Add results to the list
        models_results.append(model_info)

    return pd.DataFrame(models_results)


def get_init_args(args):
    init_args = args.get("init_args", {})
    if init_args != {}:
        return init_args

    hyper_args = args.get("hyper_parameters", {})
    if hyper_args == {}:
        raise ValueError("No hyper-parameters or init-args were provided")

    return {key: choice(value) for key, value in hyper_args.items()}
