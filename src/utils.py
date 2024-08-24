import os
import sys
import numpy as np 
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exception import CustomException
from logger import logging
from file_utils import save_object, load_object 
from src.components import data_transformation
# In your predict_pipeline.py or utils.py, add the following import
from sklearn.model_selection import RandomizedSearchCV


# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path,exist_ok=True)
#         with open(file_path, "wb") as file_obj:
#             dill.dump(obj,file_obj)
#     except Exception as e:
#         raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            logging.info(f"Training {model_name} with params: {param_grid}")

            if param_grid:  # If there are parameters to tune
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                logging.info(f"Best params for {model_name}: {gs.best_params_}")
            else:
                model.fit(X_train, y_train)  # Train model with default params
            
            assert hasattr(model, 'fit'), f"{model_name} has not been fitted properly."

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} Train Score: {train_model_score}, Test Score: {test_model_score}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return dill.load(file_obj)
    
#     except Exception as e:
#         raise CustomException(e,sys)
    
