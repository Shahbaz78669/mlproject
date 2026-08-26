import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join('artifacts', 'model_trainer.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": XGBRegressor()
            }

            params = {
                "Linear Regression": {},
                "Ridge": {
                    'alpha': [0.1, 1, 10, 100],
                },
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                },
                "ElasticNet": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                },
                "Decision Tree": {
                    'max_depth': [3, 5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Random Forest": {
                    'n_estimators': [64, 128, 256],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                "Gradient Boosting": {
                    'n_estimators': [64, 128, 256],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 8],
                    'subsample': [0.7, 0.8, 1.0],
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                },
                "Extra Trees": {
                    'n_estimators': [64, 128, 256],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan'],
                },
                "SVR": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                },
                "XGBoost": {
                    'n_estimators': [64, 128, 256],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 8],
                    'subsample': [0.7, 0.8, 1.0],
                    'colsample_bytree': [0.7, 0.8, 1.0],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )

            predicted_score = best_model.predict(X_test)
            score = r2_score(y_test, predicted_score)

            logging.info(f"Best model: {best_model_name} with R2 score: {score}")
            return score

        except Exception as e:
            raise CustomException(e, sys)