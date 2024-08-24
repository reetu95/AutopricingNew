# # src/utils/file_utils.py

# import dill
# import os
# import sys
# from src.exception import CustomException

# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path, exist_ok=True)
#         with open(file_path, "wb") as file_obj:
#             dill.dump(obj, file_obj)
#     except Exception as e:
#         raise CustomException(e, sys)

# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return dill.load(file_obj)
#     except Exception as e:
#         raise CustomException(e, sys)

# src/utils/file_utils.py

# import pickle  # Use pickle instead of dill
import os
import sys
from src.exception import CustomException
import cloudpickle as pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Use pickle
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  # Use pickle
    except Exception as e:
        raise CustomException(e, sys)
