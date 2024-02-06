import pickle

from clearml import Task
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML

from src.components.data.process_data import process_data
from src.components.data.read_dataset import read_dataset
from src.settings.metadata import PROJECT_NAME, TRAINING_TASK
from src.settings.train_params import XGBOOST_PARAMS
from src.utils.logger import logger


def train(task: Task):
    logger.info("Reading dataset...")
    df_data = read_dataset("time_series_60min_singleindex.csv")

    # Split data
    logger.info("Splitting data...")
    x_data, y_data = process_data(df_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42,
    )

    # Train model
    logger.info("Training model...")
    model = AutoML(
        results_path="out/automl",
        ml_task="multiclass_classification",
        total_time_limit=5 * 60,
        mode="Perform",
    )
    model.fit(x_train, y_train)

    print("Model score:", model.score(x_test, y_test))

    # Save to pickle
    logger.info("Saving model...")
    with open("out/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    # task = Task.init(
    #     project_name=PROJECT_NAME,
    #     task_name=TRAINING_TASK,
    # )

    # task.connect(XGBOOST_PARAMS, "XGBoost parameters")
    task = None

    train(task)
