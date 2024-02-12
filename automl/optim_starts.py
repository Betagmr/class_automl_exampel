from functools import reduce

import optuna
from clearml import Task
from optuna.trial import Trial
from sklearn.model_selection import train_test_split

from automl.base import calculate_metrics


def auto_ml(x_data, y_data, arch, metrics):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, random_state=42, test_size=0.15
    )

    for model_arch in arch:
        print(model_arch)
        score, values = optimize_strategy(x_train, y_train, x_test, y_test, model_arch, metrics)
        print(score, values)


def optimize_strategy(x_train, y_train, x_test, y_test, model_arch, metrics):
    model_init, model_args = model_arch
    model_name = model_args.get("name", model_init.__name__)
    model_hyper = model_args.get("hyper_parameters", {})
    n_trials = reduce(lambda x, y: x * y, (len(x) for x in model_hyper.values()))

    task = Task.init(
        project_name="Hyper-Parameter Optimization/automl",
        task_name=model_name,
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(model_hyper),
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            task,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            metrics=metrics,
            model_init=model_init,
            model_hyper=model_hyper,
        ),
        n_trials=n_trials,
    )

    task.close()

    return study.best_params, study.best_value


def objective(
    trial: Trial,
    task: Task,
    x_train,
    y_train,
    x_test,
    y_test,
    metrics,
    model_init,
    model_hyper: dict,
):
    params = {key: trial.suggest_categorical(key, value) for key, value in model_hyper.items()}
    model = model_init(**params)
    model.fit(x_train, y_train)

    result = model.predict(x_test)
    list_metrics = calculate_metrics(y_test, result, metrics)

    for metric, value in list_metrics.items():
        print(f"Metric {metric}: {value}")
        task.logger.report_scalar(
            title=metric,
            series="series",
            value=value,
            iteration=trial.number,
        )

    return list_metrics["accuracy"]
