import os,sys
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from imblearn.pipeline import Pipeline as ImbPipeline
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

        logging.info(f'Object successfuly saved at {file_path}')

    except Exception as e:
        logging.error(f"Failed to save object at {file_path}: {e}")
        raise CustomException(e,sys)
    
def load_object(file_path, obj):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(models: dict, param_grids: dict, X_train, y_train, X_test, y_test, preprocessor=None, resampler=None):
    """
    Evaluate multiple models with optional preprocessing, resampling, and hyperparameter tuning.

    Returns:
        results_df: DataFrame containing Accuracy, Precision, Recall, F1-score for all models
        best_models: Dictionary of fitted models (after hyperparameter tuning)
    """
    try:
        results = []
        best_models = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            # Build pipeline with preprocessor and resampler if provided
            steps = []
            if preprocessor:
                steps.append(("preprocessor", preprocessor))
            if resampler:
                steps.append(("resampler", resampler))
            steps.append(("model", model))

            pipeline = ImbPipeline(steps=steps)

            # Apply hyperparameter tuning if param grid exists
            if model_name in param_grids:
                logging.info(f"Applying GridSearchCV for {model_name}")
                grid = GridSearchCV(estimator=pipeline, param_grid=param_grids[model_name], cv=3, scoring='f1', n_jobs=-1, verbose=0)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                logging.info(f"Best params for {model_name}: {grid.best_params_}")
            else:
                pipeline.fit(X_train, y_train)
                best_model = pipeline

            # Predict on test set
            y_pred = best_model.predict(X_test)

            # Evaluate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1)
            rec = recall_score(y_test, y_pred, pos_label=1)
            f1 = f1_score(y_test, y_pred, pos_label=1)

            results.append({
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1
            })

            best_models[model_name] = best_model

        results_df = pd.DataFrame(results)
        return results_df, best_models

    except Exception as e:
        logging.error("Error in evaluating models")
        raise CustomException(e, sys) 

    

