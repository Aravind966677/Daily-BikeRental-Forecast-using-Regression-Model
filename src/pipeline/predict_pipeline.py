import sys
import os
import pandas as pd

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 season: int,
               
                 mnth: int,
                 day: int,
                 holiday: int,
                 weekday: int,
                 workingday: int,
                 weathersit: int,
                 temp: float,
                 atemp: float,
                 hum: float,
                 windspeed: float):
        self.season = season
      
        self.mnth = mnth
        self.day = day
        self.holiday = holiday
        self.weekday = weekday
        self.workingday = workingday
        self.weathersit = weathersit
        self.temp = temp
        self.atemp = atemp
        self.hum = hum
        self.windspeed = windspeed

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "season": [self.season],
                
                "mnth": [self.mnth],
                "day": [self.day],
                "holiday": [self.holiday],
                "weekday": [self.weekday],
                "workingday": [self.workingday],
                "weathersit": [self.weathersit],
                "temp": [self.temp],
                "atemp": [self.atemp],
                "hum": [self.hum],
                "windspeed": [self.windspeed]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
