import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv("notebook/data/daily_sales.csv", encoding="latin1")

            os.makedirs("artifacts", exist_ok=True)

            df["Order Date"] = pd.to_datetime(df["Order Date"])
            df = df.sort_values("Order Date")

            df.to_csv(self.config.raw_data_path, index=False)

            split_index = int(len(df) * 0.8)

            df_train = df.iloc[:split_index]
            df_test = df.iloc[split_index:]

            df_train.to_csv(self.config.train_data_path, index=False)
            df_test.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
