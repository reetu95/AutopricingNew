import sys
import pandas as pd 
import os      
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exception import CustomException
from utils import load_object
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = '/Users/reetu/Documents/Projects/AutopricingnewC2B/artifactS/model.pkl'
            preprocessor_path = '/Users/reetu/Documents/Projects/AutopricingnewC2B/artifactS/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
        self,
        brand: str,
        model: str,
        model_year: int,
        milage: str,
        fuel_type: str,
        engine: str,
        transmission: str,
        ext_col: str,
        int_col: str,
        accident: str,
        clean_title: str,
        price: str
    ):
        self.brand = brand
        self.model = model
        self.model_year = model_year
        self.milage = milage
        self.fuel_type = fuel_type
        self.engine = engine
        self.transmission = transmission
        self.ext_col = ext_col
        self.int_col = int_col
        self.accident = accident
        self.clean_title = clean_title
        self.price = price

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "brand": [self.brand],
                "model": [self.model],
                "model_year": [self.model_year],
                "milage": [self.milage],
                "fuel_type": [self.fuel_type],
                "engine": [self.engine],
                "transmission": [self.transmission],
                "ext_col": [self.ext_col],
                "int_col": [self.int_col],
                "accident": [self.accident],
                "clean_title": [self.clean_title],
                "price": [self.price]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
