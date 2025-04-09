import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformation_Config:
    preprocessor_obj_filepath = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformation_Config()

    def get_data_transformation_obj(self):
        try:
            cat_features=['gender',
                          'race_ethnicity',
                          'parental_level_of_education',
                          'lunch',
                          'test_preparation_course'
                        ]
            num_features=['reading_score',
                          'writing_score'
                          ]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('onehot', OneHotEncoder())
                ]
            )

            logging.info(f"numerical features ={num_features}")
            logging.info(f"categorical features = {cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data from csv files")

            logging.info("obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformation_obj()

            target_col_name = "math_score"

            input_train_data = train_df.drop(columns=target_col_name, axis=1)
            target_train_data = train_df[target_col_name]

            input_test_data = test_df.drop(columns=target_col_name, axis=1)
            target_test_data = test_df[target_col_name]

            logging.info("appling preprocessing on train data")
            input_train_arr = preprocessor_obj.fit_transform(input_train_data)

            logging.info("appling preprocessing on test data")
            input_test_arr = preprocessor_obj.transform(input_test_data)

            train_arr = np.c_[
                input_train_arr, np.array(target_train_data)
            ]
            test_arr = np.c_[
                input_test_arr, np.array(target_test_data)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessor_obj
            )

            logging.info("saved preprocessor object to file")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )



        except Exception as e:
            raise CustomException(e,sys)