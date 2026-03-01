import os
import sys
import pickle
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation started")

            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Convert date column
            train_df["Order Date"] = pd.to_datetime(train_df["Order Date"])
            test_df["Order Date"] = pd.to_datetime(test_df["Order Date"])

            # 🔹 DEFINE TARGET COLUMN FIRST (IMPORTANT)
            target_column = "Sales"

            # 🔹 NOW drop columns
            X_train = train_df.drop(columns=[target_column, "Order Date"])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column, "Order Date"])
            y_test = test_df[target_column]

            # Preprocessing
            preprocessor = self.get_data_transformer_object()

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            os.makedirs("artifacts", exist_ok=True)

            with open(self.config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info("Data transformation completed")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)