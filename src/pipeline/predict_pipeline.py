import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path='artifacts\model.pkl'
            model_path=os.path.join('artifcats', 'model.pkl')
            preprocessor_path='artifacts\preprocessor.pkl'
            preprocessor_path=os.path.join('artifacts', 'preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_and_column_scaled=preprocessor.transform(features)
            preds=model.predict(data_and_column_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 N:int,
                 P:int,
                 K:int,
                 temperature:float,
                 humidity:float,
                 rainfall:float,
                 ph:float):
        self.N = N
        self.P = P
        self.K = K
        self.temperature = temperature
        self.humidity = humidity
        self.rainfall = rainfall
        self.ph = ph

    def get_input_as_dataframe(self):
        try:
            custom_data_input_dict ={
                "N": [self.N],
                "P": [self.P],
                "K": [self.K],
                "temperature": [self.temperature],
                "humidity": [self.humidity],
                "rainfall": [self.rainfall],
                "ph": [self.ph]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)