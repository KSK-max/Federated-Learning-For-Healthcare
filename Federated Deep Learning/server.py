import flwr as fl
import utils
import sys
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv

import flwr as fl
from typing import Dict
import utils
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model


def fit_round(rnd: int) -> Dict:
    """Send number of training rounds to client."""
    return {"rnd": rnd}


def get_eval_fn(model: Model):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    (X_train, y_train), (X_test, y_test) = utils.load_data_client1()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_keras_model_params(model, parameters)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        train_loss, accuracy = model.evaluate(X_train, y_train, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_model = np.argmax(y_pred, axis=1)

        y_test_real = np.argmax(y_test, axis=1)

        report = classification_report(y_test_real, y_pred_model, output_dict=True)

        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']

        return loss, {
            "train_loss": train_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    return evaluate


# Start Flower server for ten rounds of federated learning
if __name__ == "__main__":
    model = utils.create_keras_model()

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=15),
    )