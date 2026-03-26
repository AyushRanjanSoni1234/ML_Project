import os
import sys

#  ✅ Fix: Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Entered the model trainer component")
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AaBoost": AdaBoostRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor()
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)

                # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
                # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("No best model found")
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            logging.info("Exception occurred in the model trainer component")
            raise CustomException(e, sys)

            