from xgboost import XGBClassifier

from src.settings.train_params import XGBOOST_PARAMS


def train_xgboost(x_train, y_train, x_test, y_test) -> XGBClassifier:
    model = XGBClassifier(**XGBOOST_PARAMS)

    return model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=True,
    )
