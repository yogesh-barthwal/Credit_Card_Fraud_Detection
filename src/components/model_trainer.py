import os, sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models, save_object
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor=None, resampler=None):
        """
        Trains multiple models with optional hyperparameter tuning and returns the best model.
        Expects train_arr and test_arr with features and target combined (last column = target).
        """
        try:
            logging.info("Starting model training")

            # Split arrays into X and y
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Define models
            models = {
                "RandomForest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
                "LGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0)
            }

            # Define param grids for GridSearchCV
            param_grids = {
                "RandomForest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]},
                "XGBoost": {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1]},
                "LGBM": {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1]},
                "CatBoost": {"model__depth": [4, 6, 8], "model__learning_rate": [0.01, 0.1], "model__iterations": [200, 400]}
            }

            # Evaluate models
            results_df, best_models = evaluate_models(
                models=models,
                param_grids=param_grids,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                preprocessor=preprocessor,
                resampler=resampler
            )

            logging.info(f"Model evaluation completed:\n{results_df}")

            # Identify best model based on F1-score
            best_model_name = results_df.sort_values("F1-score", ascending=False).iloc[0]["Model"]
            best_model = best_models[best_model_name]

            # Save the best model
            logging.info(f"Saving best model: {best_model_name}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return results_df, best_model

        except Exception as e:
            logging.error("Error during model training")
            raise CustomException(e, sys)

