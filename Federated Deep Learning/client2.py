import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_data_client2()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    model = utils.create_keras_model()

    # Define Flower client
    class HealthClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_keras_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_keras_model_params(model, parameters)
            model.fit(X_train, y_train, epochs=1, verbose=2)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_keras_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_keras_model_params(model, parameters)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=HealthClient())