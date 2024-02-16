import pandas as pd


def auto_ml(x_data, y_data, arch, metrics, strategies):
    automl_info = pd.DataFrame()

    for strategy, args in strategies:
        automl_info = strategy(x_data, y_data, arch, metrics, automl_info, **args.get("args", {}))

    return automl_info


def calculate_metrics(y_test, y_predict, metrics):
    results = {}
    for func, args in metrics:
        metric = args.get("name", func.__name__)
        result = func(y_predict, y_test, **args.get("args", {}))
        results[metric] = result

    return results
