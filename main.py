from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.train_pipeline import ModelTrainer

from src.logger import logging


if __name__ == "__main__":

    logging.info("Pipeline execution started")

    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    # Step 3: Model Training
    trainer = ModelTrainer()
    r2_score = trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    logging.info(f"Training pipeline completed. Final R2 Score: {r2_score}")
print("Fuck")