import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Model training started")

        try:
            # Split input features and target
            X_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1]
            )

            X_test, y_test = (
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Initialize model
            model = LinearRegression()

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Evaluation
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model training completed. R2 Score: {r2}")

            # Save model
            os.makedirs("artifacts", exist_ok=True)

            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
        