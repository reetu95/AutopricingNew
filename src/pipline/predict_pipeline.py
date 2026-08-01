import sys
from pathlib import Path

import pandas as pd

from src.exception import CustomException
from src.file_utils import load_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPONENTS_DIR = PROJECT_ROOT / "src" / "components"

# Compatibility for old pickled preprocessing artifacts that referenced
# data_transformation as a top-level module.
if str(COMPONENTS_DIR) not in sys.path:
    sys.path.append(str(COMPONENTS_DIR))

class PredictPipeline:
    def __init__(self):
        self.model_path = PROJECT_ROOT / "artifactS" / "model.pkl"
        self.preprocessor_path = PROJECT_ROOT / "artifactS" / "preprocessor.pkl"
        self._model = None
        self._preprocessor = None

    def _load_artifacts(self):
        if self._model is None:
            self._model = load_object(file_path=self.model_path)
        if self._preprocessor is None:
            self._preprocessor = load_object(file_path=self.preprocessor_path)

    def predict(self, features):
        try:
            self._load_artifacts()
            data_scaled = self._preprocessor.transform(features)
            preds = self._model.predict(data_scaled)
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
        clean_title: str
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
                "clean_title": [self.clean_title]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
