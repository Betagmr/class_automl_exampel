import tensorflow as tf
from keras.models import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class AutoTFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        network: Sequential,
        optimizer="adam",
        loss="categorical_crossentropy",
        n_epochs=20,
        batch_size=32,
        verbose=0,
    ):
        self.network = network
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )

    def fit(self, x, y, **kwargs):
        x_train, y_train = check_X_y(x, y, accept_sparse=True)
        self.classes_ = unique_labels(y)

        y_train = tf.one_hot(y_train, len(self.classes_)).numpy()
        y_train = y_train.reshape(-1, len(self.classes_))

        self.network.fit(
            x_train,
            y_train,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            **kwargs,
        )

        self.is_fitted = True

        return self

    def predict_proba(self, x_test):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        x_test = check_array(x_test, accept_sparse=True)

        return self.network.predict(
            x_test,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

    def predict(self, x_test):
        x_probability = self.predict_proba(x_test)

        return x_probability.argmax(axis=1)

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)

        return (y_predict == y_test).mean()
