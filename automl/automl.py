def _calculate_metrics(y_test, y_predict, metrics):
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
