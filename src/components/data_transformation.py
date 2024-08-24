# import sys
# import os
# from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.base import BaseEstimator, TransformerMixin
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from exception import CustomException
# from logger import logging
# # from utils import save_object
# from file_utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifactS', "preprocessor.pkl")

# class StringToNumericTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, columns):
#         self.columns = columns
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         X_copy = X.copy()
#         for column in self.columns:
#             X_copy[column] = X_copy[column].replace({'\$': '', ',': '', ' mi\.': ''}, regex=True).astype(float)
#         return X_copy

# class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         X_copy = X.copy()
        
#         # Calculate vehicle age
#         current_year = 2024
#         X_copy['vehicle_age'] = current_year - X_copy['model_year']
        
#         # Interaction features
#         X_copy['brand_model_year'] = X_copy['brand'] + '_' + X_copy['model_year'].astype(str)
#         X_copy['brand_milage'] = X_copy['brand'] + '_' + X_copy['milage'].astype(str)
#         X_copy['fuel_type_milage'] = X_copy['fuel_type'] + '_' + X_copy['milage'].astype(str)
#         X_copy['model_year_milage'] = X_copy['model_year'] * X_copy['milage']
#         X_copy['brand_clean_title_missing'] = X_copy['brand'] + '_' + X_copy['clean_title'].astype(str)
#         X_copy['fuel_type_clean_title_missing'] = X_copy['fuel_type'] + '_' + X_copy['clean_title'].astype(str)
#         X_copy['accident_clean_title'] = X_copy['accident'] + '_' + X_copy['clean_title'].astype(str)
#         X_copy['brand_fuel_type'] = X_copy['brand'] + '_' + X_copy['fuel_type']

#         return X_copy

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()
    
#     def get_data_transformer_object(self):
#         """
#         This function is responsible for creating the data transformation pipeline.
#         """
#         try:
#             # Define columns
#             string_to_numeric_columns = ["milage"]
#             numerical_columns = ["model_year", "milage", "vehicle_age", "model_year_milage"]
#             categorical_columns = [
#                 "brand", "model", "fuel_type", "engine", "transmission",
#                 "ext_col", "int_col", "accident", "clean_title",
#                 "brand_model_year", "brand_milage", "fuel_type_milage",
#                 "brand_clean_title_missing", "fuel_type_clean_title_missing",
#                 "accident_clean_title", "brand_fuel_type"
#             ]
            
#             # String to numeric pipeline
#             string_to_numeric_pipeline = Pipeline(
#                 steps=[
#                     ("string_to_numeric", StringToNumericTransformer(columns=string_to_numeric_columns))
#                 ]
#             )

#             # Feature engineering pipeline
#             feature_engineering_pipeline = Pipeline(
#                 steps=[
#                     ("feature_engineering", FeatureEngineeringTransformer())
#                 ]
#             )

#             # Numerical pipeline
#             num_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="median")),
#                     ("scaler", StandardScaler())
#                 ]
#             )

#             # Categorical pipeline
#             cat_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="most_frequent")),
#                     ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
#                     # ("scaler", StandardScaler(with_mean=False))
#                 ]
#             )

#             # Combine all pipelines in a ColumnTransformer
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("num_pipeline", num_pipeline, numerical_columns),
#                     ("cat_pipeline", cat_pipeline, categorical_columns)
#                 ],
#                 remainder="passthrough"
#             )
            
#             # Apply the string to numeric and feature engineering transformers before the preprocessor
#             full_pipeline = Pipeline([
#                 ("string_to_numeric", string_to_numeric_pipeline),
#                 ("feature_engineering", feature_engineering_pipeline),
#                 ("preprocessor", preprocessor)
#             ])
            
#             return full_pipeline
#         except Exception as e:
#             raise CustomException(e, sys)
    
#     def initiate_data_transformation(self, train_path, test_path):
#         try:
#             # Load the datasets
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Read train and test data completed")

#             # Remove trailing spaces in column names
#             train_df.columns = train_df.columns.str.strip()
#             test_df.columns = test_df.columns.str.strip()

#             # Convert 'price' column to numeric
#             train_df['price'] = train_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
#             test_df['price'] = test_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

#             # Apply log transformation to the price column
#             train_df['log_price'] = np.log(train_df['price'])
#             test_df['log_price'] = np.log(test_df['price'])

#             logging.info("Obtaining preprocessing object")
#             preprocessing_obj = self.get_data_transformer_object()
#             target_column_name = "log_price"  # Use log-transformed price

#             # Separate input and target features
#             input_feature_train_df = train_df.drop(columns=[target_column_name, "price"], axis=1)
#             target_feature_train_df = train_df[[target_column_name]]

#             input_feature_test_df = test_df.drop(columns=[target_column_name, "price"], axis=1)
#             target_feature_test_df = test_df[[target_column_name]]

#             # Debugging: Print the columns before transformation
#             logging.info(f"Input features before transformation (training): {input_feature_train_df.columns.tolist()}")
#             logging.info(f"Input features before transformation (testing): {input_feature_test_df.columns.tolist()}")

#             logging.info("Applying preprocessing object on training and testing dataframes")
#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

#             # Convert from sparse to dense (if applicable)
#             if hasattr(input_feature_train_arr, "toarray"):
#                 input_feature_train_arr = input_feature_train_arr.toarray()
#             if hasattr(input_feature_test_arr, "toarray"):
#                 input_feature_test_arr = input_feature_test_arr.toarray()

#             # Debugging the shapes of the arrays
#             logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
#             logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
#             logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
#             logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")

#             # Ensure the target feature has the correct shape for concatenation
#             target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1)
#             target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)

#             # Explicitly ensure both arrays are 2D and have the same dtype
#             if len(target_feature_train_df.shape) == 1:
#                 target_feature_train_df = target_feature_train_df.reshape(-1, 1)
#             if len(target_feature_test_df.shape) == 1:
#                 target_feature_test_df = target_feature_test_df.reshape(-1, 1)

#             input_feature_train_arr = np.asarray(input_feature_train_arr)
#             target_feature_train_df = np.asarray(target_feature_train_df)
#             input_feature_test_arr = np.asarray(input_feature_test_arr)
#             target_feature_test_df = np.asarray(target_feature_test_df)

#             # Additional Debugging: Ensure row count matches before concatenation
#             if input_feature_train_arr.shape[0] != target_feature_train_df.shape[0]:
#                 logging.error("Row count mismatch between input features and target feature for training data.")
#                 raise ValueError(f"Row count mismatch: input features have {input_feature_train_arr.shape[0]} rows, but target has {target_feature_train_df.shape[0]} rows.")

#             if input_feature_test_arr.shape[0] != target_feature_test_df.shape[0]:
#                 logging.error("Row count mismatch between input features and target feature for testing data.")
#                 raise ValueError(f"Row count mismatch: input features have {input_feature_test_arr.shape[0]} rows, but target has {target_feature_test_df.shape[0]} rows.")

#             # Combine the input and target arrays
#             train_arr = np.c_[
#                 input_feature_train_arr, target_feature_train_df
#             ]
#             test_arr = np.c_[
#                 input_feature_test_arr, target_feature_test_df
#             ]

#             logging.info("Saved preprocessing object")

#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path,
#             )
#         except Exception as e:
#             raise CustomException(e, sys)


import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import logging
from exception import CustomException
from src.file_utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifactS', "preprocessor.pkl")

# Define the function to convert strings to numeric values
def string_to_numeric(X):
    X = X.copy()
    for column in X.select_dtypes(include=["object"]).columns:
        try:
            X[column] = X[column].replace({'\$': '', ',': '', ' mi\.': ''}, regex=True).astype(float)
        except ValueError:
            logging.info(f"Skipping column {column} during numeric conversion. It is likely categorical.")
            continue
    return X

# Define the function to perform feature engineering
def feature_engineering(X):
    X = X.copy()
    current_year = 2024
    X['vehicle_age'] = current_year - X['model_year']

    # Interaction features
    X['brand_model_year'] = X['brand'] + '_' + X['model_year'].astype(str)
    X['brand_milage'] = X['brand'] + '_' + X['milage'].astype(str)
    X['fuel_type_milage'] = X['fuel_type'] + '_' + X['milage'].astype(str)
    X['model_year_milage'] = X['model_year'] * X['milage']
    X['brand_clean_title_missing'] = X['brand'] + '_' + X['clean_title'].astype(str)
    X['fuel_type_clean_title_missing'] = X['fuel_type'] + '_' + X['clean_title'].astype(str)
    X['accident_clean_title'] = X['accident'] + '_' + X['clean_title'].astype(str)
    X['brand_fuel_type'] = X['brand'] + '_' + X['fuel_type']

    return X

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define columns
            numerical_columns = ["model_year", "milage", "vehicle_age", "model_year_milage"]
            categorical_columns = [
                "brand", "model", "fuel_type", "engine", "transmission",
                "ext_col", "int_col", "accident", "clean_title",
                "brand_model_year", "brand_milage", "fuel_type_milage",
                "brand_clean_title_missing", "fuel_type_clean_title_missing",
                "accident_clean_title", "brand_fuel_type"
            ]

            # String to numeric pipeline
            string_to_numeric_pipeline = Pipeline(
                steps=[
                    ("string_to_numeric", FunctionTransformer(string_to_numeric))
                ]
            )

            # Feature engineering pipeline
            feature_engineering_pipeline = Pipeline(
                steps=[
                    ("feature_engineering", FunctionTransformer(feature_engineering))
                ]
            )

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            # Combine all pipelines in a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder="passthrough"
            )
            
            # Apply the string to numeric and feature engineering before the preprocessor
            full_pipeline = Pipeline([
                ("string_to_numeric", string_to_numeric_pipeline),
                ("feature_engineering", feature_engineering_pipeline),
                ("preprocessor", preprocessor)
            ])
            
            return full_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load the datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Remove trailing spaces in column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Convert 'price' column to numeric
            train_df['price'] = train_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
            test_df['price'] = test_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

            # Apply log transformation to the price column
            train_df['log_price'] = np.log(train_df['price'])
            test_df['log_price'] = np.log(test_df['price'])

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "log_price"  # Use log-transformed price

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name, "price"], axis=1)
            target_feature_train_df = train_df[[target_column_name]]

            input_feature_test_df = test_df.drop(columns=[target_column_name, "price"], axis=1)
            target_feature_test_df = test_df[[target_column_name]]

            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert from sparse to dense (if applicable)
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Ensure the target feature has the correct shape for concatenation
            target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)

            # Additional Debugging: Ensure row count matches before concatenation
            if input_feature_train_arr.shape[0] != target_feature_train_df.shape[0]:
                logging.error("Row count mismatch between input features and target feature for training data.")
                raise ValueError(f"Row count mismatch: input features have {input_feature_train_arr.shape[0]} rows, but target has {target_feature_train_df.shape[0]} rows.")

            if input_feature_test_arr.shape[0] != target_feature_test_df.shape[0]:
                logging.error("Row count mismatch between input features and target feature for testing data.")
                raise ValueError(f"Row count mismatch: input features have {input_feature_test_arr.shape[0]} rows, but target has {target_feature_test_df.shape[0]} rows.")

            # Combine the input and target arrays
            train_arr = np.c_[
                input_feature_train_arr, target_feature_train_df
            ]
            test_arr = np.c_[
                input_feature_test_arr, target_feature_test_df
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
