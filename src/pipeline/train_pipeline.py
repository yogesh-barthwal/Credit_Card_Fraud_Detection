import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    try:
        logging.info("===== Training Pipeline Started =====")

        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train: {train_path}, Test: {test_path}")

        # 2. Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
        logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

        # 3. Model Training
        trainer = ModelTrainer()
        best_model_name, best_score = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training completed. Best Model: {best_model_name} | Score: {best_score}")

        logging.info("===== Training Pipeline Finished Successfully =====")

    except Exception as e:
        logging.error("Error in Training Pipeline")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()