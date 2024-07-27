import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            # Load model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            # Print the columns of the features DataFrame for debugging
            print("Features DataFrame Columns:", features.columns)
            
            # Extract date features
            features = self.extract_date_features(features)
            
            # Print the features after extraction for debugging
            print("Features DataFrame After Date Extraction:", features.head())
            
            # Preprocess the features
            data_scaled = preprocessor.transform(features)
            
            # Print the scaled data columns if it is a DataFrame
            if isinstance(data_scaled, pd.DataFrame):
                print("Scaled Data Columns:", data_scaled.columns)
            else:
                # Print the shape or type of scaled data if it's not a DataFrame
                print("Scaled Data Shape:", data_scaled.shape)
                print("Scaled Data Type:", type(data_scaled))
            
            # Make predictions
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
    
    def extract_date_features(self, df):
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'])
            df['year'] = df['dteday'].dt.year
            df['month'] = df['dteday'].dt.month
            df['day'] = df['dteday'].dt.day
            df = df.drop(columns=['dteday'])
        return df

class CustomData:
    def __init__(self,
                 dteday: str,
                 season: int,
                 yr: int,
                 mnth: int,
                 holiday: int,
                 weekday: int,
                 workingday: int,
                 weathersit: int,
                 temp: float,
                 atemp: float,
                 hum: float,
                 windspeed: float):
        self.dteday = dteday
        self.season = season
        self.yr = yr
        self.mnth = mnth
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
                "dteday": [self.dteday],
                "season": [self.season],
                "yr": [self.yr],
                "mnth": [self.mnth],
                "holiday": [self.holiday],
                "weekday": [self.weekday],
                "workingday": [self.workingday],
                "weathersit": [self.weathersit],
                "temp": [self.temp],
                "atemp": [self.atemp],
                "hum": [self.hum],
                "windspeed": [self.windspeed]
            }
            df = pd.DataFrame(custom_data_input_dict)
            
            # Print the columns of the DataFrame for debugging
            print("Initial DataFrame Columns:", df.columns)
            
            # Drop 'dteday' if it exists in the DataFrame
            if 'dteday' in df.columns:
                df = df.drop(columns=['dteday'])
            
            # Print the DataFrame to check its contents
            print("Processed DataFrame Head:\n", df.head())
            
            return df
        except Exception as e:
            raise CustomException(e, sys)
