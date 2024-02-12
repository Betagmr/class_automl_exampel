import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def auto_ml(x_data, y_data, arch, metrics, cv=5):
    kfs = StratifiedKFold(n_splits=cv - 1, shuffle=True, random_state=42)
    split_indexes = list(kfs.split(x_data, y_data))

    models = []
    models_results = []

    for model_init, args in arch:
        model_name = args.get("name", model_init.__name__)
        y_test_list = []
        y_predict_list = []
        print("Running on model:", model_name)

        for x_train, x_test, y_train, y_test in tqdm(
            _get_split_data(x_data, y_data, split_indexes, args), total=cv - 1
        ):
            # Train the model
            model = model_init(**args.get("init_args", {}))
            model.fit(x_train, y_train, **args.get("fit_args", {}))

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
        models.append(model)
        models_results.append(model_info)

    return pd.DataFrame(models_results)


def calculate_metrics(y_test, y_predict, metrics):
    results = {}
    for func, args in metrics:
        metric = args.get("name", func.__name__)
        result = func(y_predict, y_test, **args.get("args", {}))
        results[metric] = result

    return results


def _get_split_data(x_data, y_data, split_indexes, args):
    for train_index, test_index in split_indexes:
        _x_train, _x_test = x_data[train_index], x_data[test_index]
        _y_train, _y_test = y_data[train_index], y_data[test_index]

        # Parsing the data for the model
        x_train = args.get("x_pre", lambda x: x)(_x_train)
        y_train = args.get("y_pre", lambda y: y)(_y_train)
        x_test = args.get("x_pre_t", lambda x: x)(_x_test)
        y_test = args.get("y_pre_t", lambda y: y)(_y_test)

        yield x_train, x_test, y_train, y_test
