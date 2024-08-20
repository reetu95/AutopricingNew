# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import(
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from sklearn.model_selection import GridSearchCV

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from exception import CustomException
# from logger import logging

# from src.utils import save_object, evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_traiber(self,train_array,test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:,:-1], 
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1],
#             )
#             models = {
#     "Random Forest": RandomForestRegressor(),
#     "Decision Tree": DecisionTreeRegressor(),
#     "Gradient Boosting": GradientBoostingRegressor(),
#     "Linear Regression": LinearRegression(),
#     "K-neighbors Regressor": KNeighborsRegressor(),  # Changed from "classifier" to "Regressor"
#     "XGBRegressor": XGBRegressor(),  # Corrected from "XGBClassifier"
#     "CatBoost Regressor": CatBoostRegressor(verbose=False),  # Changed from "classifier" to "Regressor"
#     "AdaBoost Regressor": AdaBoostRegressor(),  # Corrected from "Claasifier" to "Regressor"
# }


#             params = {
#                     "Decision Tree": {
#                     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                     },
#                     "Random Forest": {
#                         'n_estimators': [8, 16, 32, 64, 128, 256],
#                     },
#                     "Gradient Boosting": {
#                         'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                         'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
#                         'n_estimators': [8, 16, 32, 64, 128, 256],
#                     },
#                     "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
#                     "K-neighbors Regressor": {  # ADDED this section for K-neighbors Regressor
#                     'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to use
#                     'weights': ['uniform', 'distance'],  # Weight function used in prediction
#                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Algorithm used to compute the nearest neighbors
#                     },
#                     "XGBRegressor": {
#                     'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                     },
#                     "CatBoost Regressor": {
#                     'depth': [6, 8, 10],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'iterations': [30, 50, 100]
#                     },
#                     "AdaBoost Regressor": {
#                     'learning_rate': [0.1, 0.01, 0.5, 0.001],
#                     'n_estimators': [8, 16, 32, 64, 128, 256]
#                     }
#                     }

#             model_report = {}
#             for model_name, model in models.items():
#                 logging.info(f"Training {model_name} model.")
#                 param_grid = params.get(model_name, {})
#                 grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
#                 grid_search.fit(X_train, y_train)
                
#                 best_model = grid_search.best_estimator_
#                 model_report[model_name] = grid_search.best_score_


#             model_report:dict=evaluate_models(X_train=X_train, y_train=y_train,X_test = X_test, y_test = y_test, models=models, params= params)

#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]

#             best_model = models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found",sys)
#             logging.info(f"Best found model on both training and testing daatset")

#             save_object(
#                 file_path = self.model_trainer_config.trained_model_file_path,
#                 obj = best_model
#             )

#             predicted = best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square

#             # model_report = {}
#             # best_model = None
#             # best_score = -float('inf')

#             # for model_name, model in models.items():
#             #     logging.info(f"Training {model_name} model.")
#             #     param_grid = params.get(model_name, {})
#             #     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
#             #     grid_search.fit(X_train, y_train)
#             #     logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
#             #     model_report[model_name] = grid_search.best_score_

#             #     if grid_search.best_score_ > best_score:
#             #         best_score = grid_search.best_score_
#             #         best_model = grid_search.best_estimator_

#             # if best_score < 0.6:
#             #     raise CustomException("No best model found", sys)
            
#             # logging.info(f"Best found model: {best_model} with score: {best_score}")

#             # save_object(
#             #     file_path=self.model_trainer_config.trained_model_file_path,
#             #     obj=best_model
#             # )

#             # predicted = best_model.predict(X_test)

#             # r2_square = r2_score(y_test, predicted)
#             # return r2_square
        

#         except Exception as e:
#             print(f"Exception: {str(e)}, Sys: {sys.exc_info()}")
#             raise CustomException(str(e),sys)

"""***** 2 *************************"""

# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import(
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from sklearn.model_selection import GridSearchCV

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from exception import CustomException
# from logger import logging

# from src.utils import save_object, evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_traiber(self,train_array,test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:,:-1], 
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1],
#             )
#             models = {
#     "Random Forest": RandomForestRegressor(),
#     "Decision Tree": DecisionTreeRegressor(),
#     "Gradient Boosting": GradientBoostingRegressor(),
#     "Linear Regression": LinearRegression(),
#     "K-neighbors Regressor": KNeighborsRegressor(),  # Changed from "classifier" to "Regressor"
#     "XGBRegressor": XGBRegressor(),  # Corrected from "XGBClassifier"
#     "CatBoost Regressor": CatBoostRegressor(verbose=False),  # Changed from "classifier" to "Regressor"
#     "AdaBoost Regressor": AdaBoostRegressor(),  # Corrected from "Claasifier" to "Regressor"
# }


#             params = {
#                     "Decision Tree": {
#                     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                     },
#                     "Random Forest": {
#                         'n_estimators': [8, 16, 32, 64, 128, 256],
#                     },
#                     "Gradient Boosting": {
#                         'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                         'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
#                         'n_estimators': [8, 16, 32, 64, 128, 256],
#                     },
#                     "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
#                     "K-neighbors Regressor": {  # ADDED this section for K-neighbors Regressor
#                     'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to use
#                     'weights': ['uniform', 'distance'],  # Weight function used in prediction
#                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Algorithm used to compute the nearest neighbors
#                     },
#                     "XGBRegressor": {
#                     'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                     },
#                     "CatBoost Regressor": {
#                     'depth': [6, 8, 10],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'iterations': [30, 50, 100]
#                     },
#                     "AdaBoost Regressor": {
#                     'learning_rate': [0.1, 0.01, 0.5, 0.001],
#                     'n_estimators': [8, 16, 32, 64, 128, 256]
#                     }
#                     }

#             model_report = {}
#             for model_name, model in models.items():
#                 logging.info(f"Training {model_name} model.")
#                 param_grid = params.get(model_name, {})
#                 grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
#                 grid_search.fit(X_train, y_train)
                
#                 best_model = grid_search.best_estimator_
#                 model_report[model_name] = grid_search.best_score_


#             model_report:dict=evaluate_models(X_train=X_train, y_train=y_train,X_test = X_test, y_test = y_test, models=models, params= params)

#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]

#             best_model = models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found",sys)
#             logging.info(f"Best found model on both training and testing daatset")

#             save_object(
#                 file_path = self.model_trainer_config.trained_model_file_path,
#                 obj = best_model
#             )

#             predicted = best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square

#             # model_report = {}
#             # best_model = None
#             # best_score = -float('inf')

#             # for model_name, model in models.items():
#             #     logging.info(f"Training {model_name} model.")
#             #     param_grid = params.get(model_name, {})
#             #     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
#             #     grid_search.fit(X_train, y_train)
#             #     logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
#             #     model_report[model_name] = grid_search.best_score_

#             #     if grid_search.best_score_ > best_score:
#             #         best_score = grid_search.best_score_
#             #         best_model = grid_search.best_estimator_

#             # if best_score < 0.6:
#             #     raise CustomException("No best model found", sys)
            
#             # logging.info(f"Best found model: {best_model} with score: {best_score}")

#             # save_object(
#             #     file_path=self.model_trainer_config.trained_model_file_path,
#             #     obj=best_model
#             # )

#             # predicted = best_model.predict(X_test)

#             # r2_square = r2_score(y_test, predicted)
#             # return r2_square
        

#         except Exception as e:
#             print(f"Exception: {str(e)}, Sys: {sys.exc_info()}")
#             raise CustomException(str(e),sys)


""" ************** 3 """

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
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
from sklearn.model_selection import GridSearchCV

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
                # "K-neighbors Regressor": KNeighborsRegressor(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoost Regressor": CatBoostRegressor(verbose=False),
                # "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
    "Random Forest": {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "LGBMRegressor1": {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 10, 20],
        'subsample': [0.7, 0.8, 0.9]
    },
    "K-neighbors Regressor": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    "XGBRegressor": {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    },
    "CatBoost Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [50, 100, 200],
        'l2_leaf_reg': [1, 3, 5]
    },
    "AdaBoost Regressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'loss': ['linear', 'square', 'exponential']
    }
}


        #     model_report = {}
        #     best_model = None
        #     best_model_score = -float('inf')
        #     for model_name, model in models.items():
        #         logging.info(f"Training {model_name} model.")
        #         param_grid = params.get(model_name, {})
        #         grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
        #         logging.info(f"Gridsearch {grid_search}")
        #         grid_search.fit(X_train, y_train)
        #         logging.info(f"Grid search fitting")

        #         model_report[model_name] = grid_search.best_score_

        #         # Update best model based on the best score from GridSearchCV
        #         if grid_search.best_score_ > best_model_score:
        #             best_model_score = grid_search.best_score_
        #             best_model = grid_search.best_estimator_
            
        #     if best_model is None:
        #         raise CustomException("No best model found", sys)
            
        #     logging.info(f"Best found model: {best_model} with score: {best_model_score}")

                
        #         # best_model = grid_search.best_estimator_
        #         # model_report[model_name] = grid_search.best_score_

        #     # Evaluating all models
        #     model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)

        #     ## To get best model score from dict
        #     best_model_score = max(sorted(model_report.values()))

        #     ## To get best model name from dict
        #     best_model_name = list(model_report.keys())[
        #         list(model_report.values()).index(best_model_score)
        #     ]

        #     best_model = models[best_model_name]

        #     if best_model_score < 0.6:
        #         raise CustomException("No best model found", sys)
        #     logging.info(f"Best found model on both training and testing dataset")

        #     save_object(
        #         file_path=self.model_trainer_config.trained_model_file_path,
        #         obj=best_model
        #     )

        #     predicted = best_model.predict(X_test)

        #     # Calculate additional evaluation metrics
        #     r2_square = r2_score(y_test, predicted)
        #     mae = mean_absolute_error(y_test, predicted)
        #     mse = mean_squared_error(y_test, predicted)
        #     logging.info(f"Model Performance: R2 Score = {r2_square}, MAE = {mae}, MSE = {mse}")

        #     return r2_square

        # except Exception as e:
        #     print(f"Exception: {str(e)}, Sys: {sys.exc_info()}")
        #     raise CustomException(str(e), sys)
            # Evaluate all models using the evaluate_models function

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # Select the best model based on the highest score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

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


            predicted = best_model.predict(X_test)

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
