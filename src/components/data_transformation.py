import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exception import CustomException
from logger import logging
from utils import save_object
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")


class StringToNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].replace({'\$': '', ',': '', ' mi\.': ''}, regex=True).astype(float)
        return X_copy
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            # Convert price and milage to numeric

            string_to_numeric_columns = ["milage"]
            numerical_columns = ["model_year", "milage"]
            categorical_columns = ["brand","model","fuel_type","engine","transmission","ext_col","int_col","accident","clean_title"]
            string_to_numeric_pipeline = Pipeline(
                steps=[
                    ("string_to_numeric", StringToNumericTransformer(columns=string_to_numeric_columns))
                ]
            )
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ])
            cat_pipeline=Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorcial columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # preprocessor = ColumnTransformer(
            #     [   
            #         ("string_to_numeric", string_to_numeric_pipeline, string_to_numeric_columns),
            #         ("num_pipeline", num_pipeline, numerical_columns),
            #         ("cat_pipelines", cat_pipeline, categorical_columns)
            #     ]
            # )

            # Apply string to numeric first, then the rest
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder="passthrough"
            )
            
            # Apply the string to numeric transformer before the preprocessor
            full_pipeline = Pipeline([
                ("string_to_numeric", string_to_numeric_pipeline),
                ("preprocessor", preprocessor)
            ])
            # return preprocessor
            return full_pipeline
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Remove trailing spaces in column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Convert 'price' column to numeric by removing dollar signs and commas
            train_df['price'] = train_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
            test_df['price'] = test_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

            # Log the actual columns in the dataframe to debug issues
            logging.info(f"Columns in training dataframe: {train_df.columns.tolist()}")
            logging.info(f"Columns in testing dataframe: {test_df.columns.tolist()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "price"
            numerical_columns = ["model_year", "milage"]

            # **Change Here: Ensure target features are DataFrames**
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[[target_column_name]]  # Ensure it's a DataFrame

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[[target_column_name]]

            # Additional debugging: Print the columns before transformation
            logging.info(f"Input features before transformation (training): {input_feature_train_df.columns.tolist()}")
            logging.info(f"Input features before transformation (testing): {input_feature_test_df.columns.tolist()}")


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            # Debugging the shapes of the arrays
            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")

            # Ensure the target feature has the correct shape for concatenation
            target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)

            # Debugging the shapes after reshape
            logging.info(f"Shape of target_feature_train_df after reshape: {target_feature_train_df.shape}")
            logging.info(f"Shape of target_feature_test_df after reshape: {target_feature_test_df.shape}")

            # Print the actual arrays being concatenated
            logging.info(f"Preview of input_feature_train_arr: {input_feature_train_arr[:5]}")
            logging.info(f"Preview of target_feature_train_df: {target_feature_train_df[:5]}")

            # train_arr = np.c_[
            #     input_feature_train_arr, np.array(target_feature_train_df)
            # ]
            # test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            # Check if the number of rows matches before concatenation
            assert input_feature_train_arr.shape[0] == target_feature_train_df.shape[0], \
            f"Shape mismatch: input features have {input_feature_train_arr.shape[0]} rows but target has {target_feature_train_df.shape[0]} rows."

            assert input_feature_test_arr.shape[0] == target_feature_test_df.shape[0], \
            f"Shape mismatch: input features have {input_feature_test_arr.shape[0]} rows but target has {target_feature_test_df.shape[0]} rows."


            train_arr = np.c_[
            input_feature_train_arr, target_feature_train_df
            ]
            test_arr = np.c_[
            input_feature_test_arr, target_feature_test_df
            ]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
