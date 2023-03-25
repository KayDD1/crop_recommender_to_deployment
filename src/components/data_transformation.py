import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer,RobustScaler
from scipy.stats import boxcox

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function handles the data transformation 
        '''
        try:
            features=['N','P','K','ph','temperature','humidity','rainfall']
            boxcox_column=['K']

            #create the Pipelines for boxcox, outlier remover and scaling
            column_power_transform_pipeline=Pipeline(
                steps=[
                    #apply BoxCox transformation on K column
                                ('pt',PowerTransformer())
                ]
            )
            data_preparation_pipeline=Pipeline(
                steps=[
                                #remove outlier from the dataset
                                ('rs',RobustScaler()), 
                                #apply StandardScaler on the dataset
                                ('ss',StandardScaler())
                                ])

            logging.info("Transformed the K column")
            logging.info("Removed Outliers from the Dataset")
            logging.info("Normalize the dataset using StandardScaler")
            preprocessor = ColumnTransformer(
                [
                ("column_boxcox_pipeline", column_power_transform_pipeline, boxcox_column),
                ("data_preparation_pipeline", data_preparation_pipeline, features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loading train and test data completed")
            
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "label"
            features = ['N','P','K','ph','temperature','humidity','rainfall']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying Preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved Preprocessing objects.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)