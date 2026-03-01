import sys
import os
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            logging.info("Loading model and preprocessor")

            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            with open(self.preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            data_scaled = preprocessor.transform(features)

            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, **kwargs):
        self.input_dict = kwargs

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame([self.input_dict])
        except Exception as e:
            raise CustomException(e, sys)