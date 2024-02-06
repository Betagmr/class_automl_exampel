from typing import TypedDict


class XGBoostParams(TypedDict):
    n_estimators: int
    early_stopping_rounds: int
    learning_rate: float
    colsample_bytree: float
    max_depth: int


XGBOOST_PARAMS: XGBoostParams = {
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "learning_rate": 0.01,
    "colsample_bytree": 0.8,
    "max_depth": 7,
}
