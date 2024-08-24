# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import(
#     AdaBoostRegressor,
#     RandomForestRegressor,
# )
# from sklearn.exceptions import NotFittedError
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.model_selection import GridSearchCV

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from exception import CustomException
# from logger import logging

# from src.utils import save_object, evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1], 
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1],
#             )

#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "LGBMRegressor1": LGBMRegressor(num_leaves=31, learning_rate=0.1, n_estimators=100),
#                 "LGBMRegressor2": LGBMRegressor(num_leaves=50, learning_rate=0.05, n_estimators=200),
#                 "LGBMRegressor3": LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=150, subsample=0.8),
#                 "Linear Regression": LinearRegression(),
#                 # "K-neighbors Regressor": KNeighborsRegressor(),
#                 # "XGBRegressor": XGBRegressor(),
#                 # "CatBoost Regressor": CatBoostRegressor(verbose=False),
#                 # "AdaBoost Regressor": AdaBoostRegressor(),
#             }

#             params = {
#     "Random Forest": {
#         'n_estimators': [10, 50, 100, 200],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['auto', 'sqrt', 'log2']
#     },
#     "Decision Tree": {
#         'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#         'splitter': ['best', 'random'],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     },
#     "LGBMRegressor1": {
#         'num_leaves': [31, 50, 100],
#         'learning_rate': [0.05, 0.1, 0.2],
#         'n_estimators': [50, 100, 200],
#         'max_depth': [-1, 10, 20],
#         'subsample': [0.7, 0.8, 0.9]
#     },
#     "K-neighbors Regressor": {
#         'n_neighbors': [3, 5, 7, 10],
#         'weights': ['uniform', 'distance'],
#         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
#     },
#     "XGBRegressor": {
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [50, 100, 200],
#         'max_depth': [3, 5, 10],
#         'subsample': [0.7, 0.8, 0.9],
#         'colsample_bytree': [0.7, 0.8, 0.9]
#     },
#     "CatBoost Regressor": {
#         'depth': [6, 8, 10],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'iterations': [50, 100, 200],
#         'l2_leaf_reg': [1, 3, 5]
#     },
#     "AdaBoost Regressor": {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 1],
#         'loss': ['linear', 'square', 'exponential']
#     }
# }


#         #     model_report = {}
#         #     best_model = None
#         #     best_model_score = -float('inf')
#         #     for model_name, model in models.items():
#         #         logging.info(f"Training {model_name} model.")
#         #         param_grid = params.get(model_name, {})
#         #         grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
#         #         logging.info(f"Gridsearch {grid_search}")
#         #         grid_search.fit(X_train, y_train)
#         #         logging.info(f"Grid search fitting")

#         #         model_report[model_name] = grid_search.best_score_

#         #         # Update best model based on the best score from GridSearchCV
#         #         if grid_search.best_score_ > best_model_score:
#         #             best_model_score = grid_search.best_score_
#         #             best_model = grid_search.best_estimator_
            
#         #     if best_model is None:
#         #         raise CustomException("No best model found", sys)
            
#         #     logging.info(f"Best found model: {best_model} with score: {best_model_score}")

                
#         #         # best_model = grid_search.best_estimator_
#         #         # model_report[model_name] = grid_search.best_score_

#         #     # Evaluating all models
#         #     model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)

#         #     ## To get best model score from dict
#         #     best_model_score = max(sorted(model_report.values()))

#         #     ## To get best model name from dict
#         #     best_model_name = list(model_report.keys())[
#         #         list(model_report.values()).index(best_model_score)
#         #     ]

#         #     best_model = models[best_model_name]

#         #     if best_model_score < 0.6:
#         #         raise CustomException("No best model found", sys)
#         #     logging.info(f"Best found model on both training and testing dataset")

#         #     save_object(
#         #         file_path=self.model_trainer_config.trained_model_file_path,
#         #         obj=best_model
#         #     )

#         #     predicted = best_model.predict(X_test)

#         #     # Calculate additional evaluation metrics
#         #     r2_square = r2_score(y_test, predicted)
#         #     mae = mean_absolute_error(y_test, predicted)
#         #     mse = mean_squared_error(y_test, predicted)
#         #     logging.info(f"Model Performance: R2 Score = {r2_square}, MAE = {mae}, MSE = {mse}")

#         #     return r2_square

#         # except Exception as e:
#         #     print(f"Exception: {str(e)}, Sys: {sys.exc_info()}")
#         #     raise CustomException(str(e), sys)
#             # Evaluate all models using the evaluate_models function

#             model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

#             # Select the best model based on the highest score
#             best_model_score = max(sorted(model_report.values()))
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found", sys)

#             logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             # Add a check before predicting to ensure the model is fitted
#             try:
#                 predicted = best_model.predict(X_test)
#             except NotFittedError as e:
#                 logging.error(f"Model {best_model_name} is not fitted properly.")
#                 raise CustomException(f"Model {best_model_name} is not fitted properly.", sys)


#             predicted = best_model.predict(X_test)

#             # Calculate additional evaluation metrics
#             r2_square = r2_score(y_test, predicted)
#             mae = mean_absolute_error(y_test, predicted)
#             mse = mean_squared_error(y_test, predicted)
#             logging.info(f"Model Performance: R2 Score = {r2_square}, MAE = {mae}, MSE = {mse}")

#             return r2_square

#         except NotFittedError as e:
#             logging.error(f"Model {best_model_name} is not fitted properly: {str(e)}")
#             raise CustomException(f"Model {best_model_name} is not fitted properly.", sys)
#         except Exception as e:
#             logging.error(f"An error occurred: {str(e)}")
#             raise CustomException(f"An unexpected error occurred: {str(e)}", sys)

""" This is working """

# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     RandomForestRegressor,
# )
# from sklearn.exceptions import NotFittedError
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from exception import CustomException
# from logger import logging

# from src.utils import save_object, evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1], 
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1],
#             )

#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "LGBMRegressor1": LGBMRegressor(num_leaves=31, learning_rate=0.1, n_estimators=100),
#                 "LGBMRegressor2": LGBMRegressor(num_leaves=50, learning_rate=0.05, n_estimators=200),
#                 "LGBMRegressor3": LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=150, subsample=0.8),
#                 "Linear Regression": LinearRegression(),
#                 "K-neighbors Regressor": KNeighborsRegressor(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoost Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#             }

#             params = {
#                 "Random Forest": {
#                     'n_estimators': [100, 200],
#                     'max_depth': [10, 20],
#                     'min_samples_split': [2, 5],
#                     'min_samples_leaf': [1, 2]
#                 },
#                 "Decision Tree": {
#                     'max_depth': [10, 20, None],
#                     'min_samples_split': [2, 5],
#                     'min_samples_leaf': [1, 2]
#                 },
#                 # "XGBRegressor": {
#                 #     'learning_rate': [0.01, 0.1],
#                 #     'n_estimators': [100, 200],
#                 #     'max_depth': [3, 5],
#                 #     'subsample': [0.8, 1.0],
#                 # },
#                 "CatBoost Regressor": {
#                     'iterations': [100, 200],
#                     'learning_rate': [0.03, 0.1],
#                     'depth': [6, 10],
#                 },
#                 "AdaBoost Regressor": {
#                     'n_estimators': [50, 100],
#                     'learning_rate': [0.01, 0.1],
#                 }
#             }

#             model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

#             # Select the best model based on the highest score
#             best_model_score = max(sorted(model_report.values()))
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found", sys)

#             logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             # Add a check before predicting to ensure the model is fitted
#             try:
#                 predicted = best_model.predict(X_test)
#             except NotFittedError as e:
#                 logging.error(f"Model {best_model_name} is not fitted properly.")
#                 raise CustomException(f"Model {best_model_name} is not fitted properly.", sys)

#             # Calculate additional evaluation metrics
#             r2_square = r2_score(y_test, predicted)
#             mae = mean_absolute_error(y_test, predicted)
#             mse = mean_squared_error(y_test, predicted)
#             logging.info(f"Model Performance: R2 Score = {r2_square}, MAE = {mae}, MSE = {mse}")

#             return r2_square

#         except NotFittedError as e:
#             logging.error(f"Model {best_model_name} is not fitted properly: {str(e)}")
#             raise CustomException(f"Model {best_model_name} is not fitted properly.", sys)
#         except Exception as e:
#             logging.error(f"An error occurred: {str(e)}")
#             raise CustomException(f"An unexpected error occurred: {str(e)}", sys)


import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exception import CustomException
from logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "LGBMRegressor1": LGBMRegressor(num_leaves=31, learning_rate=0.1, n_estimators=100),
                "LGBMRegressor2": LGBMRegressor(num_leaves=50, learning_rate=0.05, n_estimators=200),
                "LGBMRegressor3": LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=150, subsample=0.8),
                "Linear Regression": LinearRegression(),
                "K-neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Decision Tree": {
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                # "XGBRegressor": {
                #     'learning_rate': [0.01, 0.1],
                #     'n_estimators': [100, 200],
                #     'max_depth': [3, 5],
                #     'subsample': [0.8, 1.0],
                # },
                "CatBoost Regressor": {
                    'iterations': [100, 200],
                    'learning_rate': [0.03, 0.1],
                    'depth': [6, 10],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # Filter out models with significant overfitting
            filtered_model_report = {}
            for model_name in model_report.keys():
                model = models[model_name]
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                if abs(train_score - test_score) <= 0.2:  # Example threshold for overfitting
                    filtered_model_report[model_name] = test_score
                else:
                    logging.info(f"{model_name} is excluded due to overfitting (Train-Test Diff: {train_score - test_score:.3f})")

            # Select the best model based on the highest test score
            if not filtered_model_report:
                raise CustomException("No suitable model found after excluding overfitted models", sys)

            best_model_score = max(filtered_model_report.values())
            best_model_name = list(filtered_model_report.keys())[
                list(filtered_model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Add a check before predicting to ensure the model is fitted
            try:
                predicted = best_model.predict(X_test)
            except NotFittedError as e:
                logging.error(f"Model {best_model_name} is not fitted properly.")
                raise CustomException(f"Model {best_model_name} is not fitted properly.", sys)

            # Calculate additional evaluation metrics
            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            logging.info(f"Model Performance: R2 Score = {r2_square}, MAE = {mae}, MSE = {mse}")

            return r2_square

        except NotFittedError as e:
            logging.error(f"Model {best_model_name} is not fitted properly: {str(e)}")
            raise CustomException(f"Model {best_model_name} is not fitted properly.", sys)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise CustomException(f"An unexpected error occurred: {str(e)}", sys)
